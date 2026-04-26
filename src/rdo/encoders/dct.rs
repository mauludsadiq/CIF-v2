//! dct_tile encoder.
//!
//! 32x32 full-tile DCT-II per channel. Keeps first K zigzag coefficients.
//! Payload: 2 (k: u16) + 3 * K * 2 (i16 coefficients) bytes.
//! Rate: payload_len * 8 bits.
//!
//! Quantization: q_step = max(1, round(BASE / sqrt(quality_lambda_normalized)))
//! K = 32 per channel (fixed in v0).

use crate::rdo::types::{EncodedRegion, LmsTile, RegionEncoder, TILE_SIZE, FIX_SCALE};

const K: usize = 16; // coefficients per channel — payload = 1 + 3*16*2 = 97 bytes = 776 bits
const BASE: f64 = 4096.0; // base quantization divisor

pub struct DctTile {
    pub quality_lambda: i64,
}

impl DctTile {
    pub fn new(quality_lambda: i64) -> Self {
        Self { quality_lambda }
    }

    fn q_step(&self) -> i64 {
        let q_norm = (self.quality_lambda as f64 / FIX_SCALE as f64).max(1.0);
        let q_step = (FIX_SCALE as f64 / (BASE * q_norm.sqrt())).round() as i64;
        q_step.max(1)
    }
}

/// Build zigzag index order for an NxN matrix.
/// Returns (row, col) pairs in zigzag scan order.
fn zigzag_indices(n: usize) -> Vec<(usize, usize)> {
    let mut result = Vec::with_capacity(n * n);
    let mut r = 0usize;
    let mut c = 0usize;
    let mut going_up = true;
    for _ in 0..n*n {
        result.push((r, c));
        if going_up {
            if c == n-1 { r += 1; going_up = false; }
            else if r == 0 { c += 1; going_up = false; }
            else { r -= 1; c += 1; }
        } else {
            if r == n-1 { c += 1; going_up = true; }
            else if c == 0 { r += 1; going_up = true; }
            else { r += 1; c -= 1; }
        }
    }
    result
}

/// Forward DCT-II on a 1D array of length N.
/// Uses the standard definition: X[k] = sum_{n=0}^{N-1} x[n] * cos(pi*(n+0.5)*k/N)
fn dct1d(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0f64; n];
    for k in 0..n {
        let mut sum = 0f64;
        for i in 0..n {
            sum += x[i] * ((std::f64::consts::PI * (i as f64 + 0.5) * k as f64) / n as f64).cos();
        }
        out[k] = sum;
    }
    out
}

/// Inverse DCT-II (DCT-III) on a 1D array of length N.
fn idct1d(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0f64; n];
    for i in 0..n {
        let mut sum = x[0] / 2.0;
        for k in 1..n {
            sum += x[k] * ((std::f64::consts::PI * (i as f64 + 0.5) * k as f64) / n as f64).cos();
        }
        out[i] = sum * 2.0 / n as f64;
    }
    out
}

/// 2D DCT-II: apply 1D DCT along rows then columns.
fn dct2d(plane: &[f64], n: usize) -> Vec<f64> {
    // DCT along rows
    let mut tmp = vec![0f64; n * n];
    for r in 0..n {
        let row: Vec<f64> = (0..n).map(|c| plane[r*n+c]).collect();
        let dct_row = dct1d(&row);
        for c in 0..n { tmp[r*n+c] = dct_row[c]; }
    }
    // DCT along columns
    let mut out = vec![0f64; n * n];
    for c in 0..n {
        let col: Vec<f64> = (0..n).map(|r| tmp[r*n+c]).collect();
        let dct_col = dct1d(&col);
        for r in 0..n { out[r*n+c] = dct_col[r]; }
    }
    out
}

/// 2D inverse DCT: apply along columns then rows.
fn idct2d(coeffs: &[f64], n: usize) -> Vec<f64> {
    // IDCT along columns
    let mut tmp = vec![0f64; n * n];
    for c in 0..n {
        let col: Vec<f64> = (0..n).map(|r| coeffs[r*n+c]).collect();
        let idct_col = idct1d(&col);
        for r in 0..n { tmp[r*n+c] = idct_col[r]; }
    }
    // IDCT along rows
    let mut out = vec![0f64; n * n];
    for r in 0..n {
        let row: Vec<f64> = (0..n).map(|c| tmp[r*n+c]).collect();
        let idct_row = idct1d(&row);
        for c in 0..n { out[r*n+c] = idct_row[c]; }
    }
    out
}

impl RegionEncoder for DctTile {
    fn name(&self) -> &'static str { "dct_tile" }

    fn encode(&self, tile: &LmsTile) -> EncodedRegion {
        let n = TILE_SIZE;
        let zz = zigzag_indices(n);
        let qs = self.q_step();
        let mut payload = Vec::new();

        // Header: k as u8 (1 byte) + q_step as i64 (8 bytes)
        payload.push(K as u8);
        payload.extend_from_slice(&qs.to_le_bytes());

        for c in 0..3 {
            let plane: Vec<f64> = (0..n*n).map(|i| {
                let x = i % n; let y = i / n;
                tile.get(x, y, c) as f64 / FIX_SCALE as f64
            }).collect();

            let dct = dct2d(&plane, n);
            let norm = (n * n) as f64; // normalize so coefficients are in [0,1] range

            for i in 0..K {
                let (r, col) = zz[i];
                let v = dct[r*n+col] / norm;  // normalized coefficient
                let q = (v / qs as f64 * FIX_SCALE as f64).round() as i64;
                let q16 = q.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
                payload.extend_from_slice(&q16.to_le_bytes());
            }
        }

        let rate_bits = (payload.len() * 8) as u64;
        EncodedRegion {
            encoder_name: self.name(),
            rate_bits,
            decode_cost: (n * n * 3 * K) as u64,
            encode_cost: (n * n * 3 * K * 4) as u64,
            memory_cost: payload.len() as u64,
            replay_cost: (n * n * 3 * K * 4) as u64,
            payload,
        }
    }

    fn decode(&self, encoded: &EncodedRegion, out: &mut LmsTile) {
        let p = &encoded.payload;
        let k = p[0] as usize;
        let q_step = i64::from_le_bytes(p[1..9].try_into().unwrap());
        let p = &p[9..]; // skip header
        let n = TILE_SIZE;
        let zz = zigzag_indices(n);

        for c in 0..3 {
            let base = c * k * 2;
            let mut dct = vec![0f64; n * n];

            let norm = (n * n) as f64;
            for i in 0..k {
                let q16 = i16::from_le_bytes(p[base+i*2..base+i*2+2].try_into().unwrap());
                let v = q16 as f64 * q_step as f64 / FIX_SCALE as f64 * norm; // denormalize
                let (r, col) = zz[i];
                dct[r*n+col] = v;
            }

            let plane = idct2d(&dct, n);

            for y in 0..n {
                for x in 0..n {
                    let v = (plane[y*n+x] * FIX_SCALE as f64).round() as i64;
                    out.set(x, y, c, v);
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
