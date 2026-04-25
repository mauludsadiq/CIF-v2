//! wavelet_tile encoder.
//!
//! Single-level 2D Haar transform per channel on a 32×32 LMS tile.
//! Coefficients are quantized and zero-RLE encoded.
//! Payload is variable length. rate_bits = payload.len() * 8.

use crate::rdo::types::{EncodedRegion, LmsTile, RegionEncoder, TILE_SIZE, FIX_SCALE};

const HALF: usize = TILE_SIZE / 2; // 16
const Q_STEP: i64 = FIX_SCALE / 256; // quantization step ~3906

pub struct WaveletTile;

/// Forward 2D Haar on a TILE_SIZE×TILE_SIZE plane (row-major, fixed-point).
/// Returns [LL, LH, HL, HH] each of size HALF×HALF.
fn haar_forward(plane: &[i64]) -> ([i64; HALF*HALF], [i64; HALF*HALF], [i64; HALF*HALF], [i64; HALF*HALF]) {
    let mut ll = [0i64; HALF*HALF];
    let mut lh = [0i64; HALF*HALF];
    let mut hl = [0i64; HALF*HALF];
    let mut hh = [0i64; HALF*HALF];

    for hy in 0..HALF {
        for hx in 0..HALF {
            let a = plane[(2*hy)  *TILE_SIZE + (2*hx)  ];
            let b = plane[(2*hy)  *TILE_SIZE + (2*hx+1)];
            let c = plane[(2*hy+1)*TILE_SIZE + (2*hx)  ];
            let d = plane[(2*hy+1)*TILE_SIZE + (2*hx+1)];
            let idx = hy * HALF + hx;
            ll[idx] = (a + b + c + d) / 4;
            lh[idx] = (a - b + c - d) / 4; // horizontal detail
            hl[idx] = (a + b - c - d) / 4; // vertical detail
            hh[idx] = (a - b - c + d) / 4; // diagonal detail
        }
    }
    (ll, lh, hl, hh)
}

/// Inverse 2D Haar — reconstruct TILE_SIZE×TILE_SIZE from four subbands.
fn haar_inverse(ll: &[i64], lh: &[i64], hl: &[i64], hh: &[i64]) -> [i64; TILE_SIZE*TILE_SIZE] {
    let mut out = [0i64; TILE_SIZE*TILE_SIZE];
    for hy in 0..HALF {
        for hx in 0..HALF {
            let idx = hy * HALF + hx;
            let a = ll[idx] + lh[idx] + hl[idx] + hh[idx];
            let b = ll[idx] - lh[idx] + hl[idx] - hh[idx];
            let c = ll[idx] + lh[idx] - hl[idx] - hh[idx];
            let d = ll[idx] - lh[idx] - hl[idx] + hh[idx];
            out[(2*hy)  *TILE_SIZE + (2*hx)  ] = a;
            out[(2*hy)  *TILE_SIZE + (2*hx+1)] = b;
            out[(2*hy+1)*TILE_SIZE + (2*hx)  ] = c;
            out[(2*hy+1)*TILE_SIZE + (2*hx+1)] = d;
        }
    }
    out
}

/// Quantize: divide by Q_STEP and round.
fn quantize(v: i64) -> i16 {
    (v / Q_STEP).clamp(i16::MIN as i64, i16::MAX as i64) as i16
}

/// Dequantize: multiply by Q_STEP.
fn dequantize(v: i16) -> i64 {
    v as i64 * Q_STEP
}

/// RLE encode a slice of i16 values.
/// Format: (value: i16, count: u16) pairs, little-endian.
/// Zeros are run-length encoded; non-zeros are emitted with count=1.
fn rle_encode(vals: &[i16]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < vals.len() {
        if vals[i] == 0 {
            let mut count = 1u16;
            loop {
                let next = i + count as usize;
                if next >= vals.len() || vals[next] != 0 || count == u16::MAX { break; }
                count += 1;
            }
            out.extend_from_slice(&0i16.to_le_bytes());
            out.extend_from_slice(&count.to_le_bytes());
            i += count as usize;
        } else {
            out.extend_from_slice(&vals[i].to_le_bytes());
            out.extend_from_slice(&1u16.to_le_bytes());
            i += 1;
        }
    }
    out
}

/// RLE decode back to i16 values.
fn rle_decode(data: &[u8], expected: usize) -> Vec<i16> {
    let mut out = Vec::with_capacity(expected);
    let mut i = 0;
    while i + 3 < data.len() {
        let val = i16::from_le_bytes([data[i], data[i+1]]);
        let count = u16::from_le_bytes([data[i+2], data[i+3]]) as usize;
        for _ in 0..count { out.push(val); }
        i += 4;
    }
    out
}

impl RegionEncoder for WaveletTile {
    fn name(&self) -> &'static str { "wavelet_tile" }

    fn encode(&self, tile: &LmsTile) -> EncodedRegion {
        let mut payload = Vec::new();

        for c in 0..3 {
            let plane: [i64; TILE_SIZE*TILE_SIZE] = {
                let mut p = [0i64; TILE_SIZE*TILE_SIZE];
                for y in 0..TILE_SIZE {
                    for x in 0..TILE_SIZE {
                        p[y*TILE_SIZE+x] = tile.get(x, y, c);
                    }
                }
                p
            };

            let (ll, lh, hl, hh) = haar_forward(&plane);

            // Quantize all subbands — LL uses finer step (divide by 1 less)
            let mut coeffs = Vec::with_capacity(4 * HALF * HALF);
            for &v in &ll { coeffs.push(quantize(v)); }
            for &v in &lh { coeffs.push(quantize(v)); }
            for &v in &hl { coeffs.push(quantize(v)); }
            for &v in &hh { coeffs.push(quantize(v)); }

            let encoded = rle_encode(&coeffs);
            // Write channel length prefix (u32 LE) then data
            payload.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
            payload.extend_from_slice(&encoded);
        }

        let rate_bits = (payload.len() * 8) as u64;
        EncodedRegion {
            encoder_name: self.name(),
            rate_bits,
            decode_cost: (TILE_SIZE * TILE_SIZE * 3 * 8) as u64,
            encode_cost: (TILE_SIZE * TILE_SIZE * 3 * 8) as u64,
            memory_cost: payload.len() as u64,
            replay_cost: (TILE_SIZE * TILE_SIZE * 3 * 8) as u64,
            payload,
        }
    }

    fn decode(&self, encoded: &EncodedRegion, out: &mut LmsTile) {
        let p = &encoded.payload;
        let n_coeffs = 4 * HALF * HALF;
        let mut offset = 0usize;

        for c in 0..3 {
            let len = u32::from_le_bytes(p[offset..offset+4].try_into().unwrap()) as usize;
            offset += 4;
            let coeffs = rle_decode(&p[offset..offset+len], n_coeffs);
            offset += len;

            let mut ll = [0i64; HALF*HALF];
            let mut lh = [0i64; HALF*HALF];
            let mut hl = [0i64; HALF*HALF];
            let mut hh = [0i64; HALF*HALF];

            for i in 0..HALF*HALF {
                ll[i] = dequantize(*coeffs.get(i).unwrap_or(&0));
                lh[i] = dequantize(*coeffs.get(HALF*HALF + i).unwrap_or(&0));
                hl[i] = dequantize(*coeffs.get(2*HALF*HALF + i).unwrap_or(&0));
                hh[i] = dequantize(*coeffs.get(3*HALF*HALF + i).unwrap_or(&0));
            }

            let recon = haar_inverse(&ll, &lh, &hl, &hh);
            for y in 0..TILE_SIZE {
                for x in 0..TILE_SIZE {
                    out.set(x, y, c, recon[y*TILE_SIZE+x]);
                }
            }
        }
    }

    fn rate_bits(&self, e: &EncodedRegion) -> u64 { e.rate_bits }
    fn decode_cost(&self, e: &EncodedRegion) -> u64 { e.decode_cost }
    fn encode_cost(&self, e: &EncodedRegion) -> u64 { e.encode_cost }
    fn memory_cost(&self, e: &EncodedRegion) -> u64 { e.memory_cost }
    fn replay_cost(&self, e: &EncodedRegion) -> u64 { e.replay_cost }
}
