//! affine_lms encoder.
//!
//! Payload: 72 bytes — 3 channels × [offset, dx, dy] × i64 little-endian, scale FIX_SCALE.
//! Fits a linear model: pixel(x,y)[c] = offset[c] + dx[c]*x + dy[c]*y
//! rate_bits: 576

use crate::rdo::types::{EncodedRegion, LmsTile, RegionEncoder, TILE_SIZE, FIX_SCALE};

pub struct AffineLms;

impl RegionEncoder for AffineLms {
    fn name(&self) -> &'static str { "affine_lms" }

    fn encode(&self, tile: &LmsTile) -> EncodedRegion {
        let n = (TILE_SIZE * TILE_SIZE) as f64;
        let mut payload = Vec::with_capacity(72);

        for c in 0..3 {
            // Least-squares fit: minimise sum((v - offset - dx*x - dy*y)^2)
            // Normal equations for [offset, dx, dy]:
            // [n,    sx,   sy  ] [offset]   [sv  ]
            // [sx,   sx2,  sxy ] [dx    ] = [svx ]
            // [sy,   sxy,  sy2 ] [dy    ]   [svy ]
            let mut sv = 0f64; let mut svx = 0f64; let mut svy = 0f64;
            let mut sx = 0f64; let mut sy = 0f64;
            let mut sx2 = 0f64; let mut sy2 = 0f64; let mut sxy = 0f64;

            for y in 0..TILE_SIZE {
                for x in 0..TILE_SIZE {
                    let v = tile.get(x, y, c) as f64 / FIX_SCALE as f64;
                    let xf = x as f64; let yf = y as f64;
                    sv += v; svx += v*xf; svy += v*yf;
                    sx += xf; sy += yf;
                    sx2 += xf*xf; sy2 += yf*yf; sxy += xf*yf;
                }
            }

            // Solve 3×3 system via Cramer's rule
            // Simplified: centre coordinates to reduce numerical error
            let cx = (TILE_SIZE - 1) as f64 / 2.0;
            let cy = (TILE_SIZE - 1) as f64 / 2.0;

            // Recompute with centred coords
            let mut sv2 = 0f64; let mut svx2 = 0f64; let mut svy2 = 0f64;
            let mut sxx = 0f64; let mut syy = 0f64; let mut sxyc = 0f64;

            for y in 0..TILE_SIZE {
                for x in 0..TILE_SIZE {
                    let v = tile.get(x, y, c) as f64 / FIX_SCALE as f64;
                    let xf = x as f64 - cx; let yf = y as f64 - cy;
                    sv2 += v; svx2 += v*xf; svy2 += v*yf;
                    sxx += xf*xf; syy += yf*yf; sxyc += xf*yf;
                }
            }

            // With centred coords, cross terms vanish if grid is symmetric
            let det_xx = n * sxx - 0.0 * 0.0; // sx_c = 0 by centering
            let dx = if det_xx.abs() > 1e-10 { svx2 / sxx } else { 0.0 };
            let det_yy = n * syy - 0.0 * 0.0;
            let dy = if det_yy.abs() > 1e-10 { svy2 / syy } else { 0.0 };
            let offset_c = sv2 / n; // mean value at centre

            // Convert back to origin: offset_origin = offset_c - dx*cx - dy*cy
            let offset = offset_c - dx * cx - dy * cy;

            let to_fixed = |v: f64| -> i64 { (v * FIX_SCALE as f64).round() as i64 };
            payload.extend_from_slice(&to_fixed(offset).to_le_bytes());
            payload.extend_from_slice(&to_fixed(dx).to_le_bytes());
            payload.extend_from_slice(&to_fixed(dy).to_le_bytes());
        }

        EncodedRegion {
            encoder_name: self.name(),
            rate_bits: 576,
            decode_cost: (TILE_SIZE * TILE_SIZE * 3 * 3) as u64,
            encode_cost: (TILE_SIZE * TILE_SIZE * 3 * 10) as u64,
            memory_cost: 72,
            replay_cost: (TILE_SIZE * TILE_SIZE * 3 * 10) as u64,
            payload,
        }
    }

    fn decode(&self, encoded: &EncodedRegion, out: &mut LmsTile) {
        let p = &encoded.payload;
        let mut coeffs = [[0i64; 3]; 3]; // [channel][offset, dx, dy]
        for c in 0..3 {
            let base = c * 24;
            coeffs[c][0] = i64::from_le_bytes(p[base..base+8].try_into().unwrap());
            coeffs[c][1] = i64::from_le_bytes(p[base+8..base+16].try_into().unwrap());
            coeffs[c][2] = i64::from_le_bytes(p[base+16..base+24].try_into().unwrap());
        }
        for y in 0..TILE_SIZE {
            for x in 0..TILE_SIZE {
                for c in 0..3 {
                    let v = coeffs[c][0]
                        + coeffs[c][1] * x as i64
                        + coeffs[c][2] * y as i64;
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
