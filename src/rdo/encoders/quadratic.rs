//! quadratic_lms encoder.
//!
//! Payload: 144 bytes — 3 channels × 6 coefficients × i64 little-endian, scale FIX_SCALE.
//! Fits: pixel(x,y)[c] = a0 + a1*x + a2*y + a3*x² + a4*y² + a5*x*y
//! rate_bits: 576  (3 channels × 6 f32 coefficients = 72 bytes)

use crate::rdo::types::{EncodedRegion, LmsTile, RegionEncoder, TILE_SIZE, FIX_SCALE};

pub struct QuadraticLms;

impl RegionEncoder for QuadraticLms {
    fn name(&self) -> &'static str { "quadratic_lms" }

    fn encode(&self, tile: &LmsTile) -> EncodedRegion {
        let mut payload = Vec::with_capacity(72); // 6 f32 × 3 channels
        let cx = (TILE_SIZE - 1) as f64 / 2.0;
        let cy = (TILE_SIZE - 1) as f64 / 2.0;

        for c in 0..3 {
            // Build 6×6 normal equations for [a0,a1,a2,a3,a4,a5]
            // basis = [1, x, y, x², y², x*y] with centred coords
            let mut ata = [[0f64; 6]; 6];
            let mut atb = [0f64; 6];

            for y in 0..TILE_SIZE {
                for x in 0..TILE_SIZE {
                    let v = tile.get(x, y, c) as f64 / FIX_SCALE as f64;
                    let xf = x as f64 - cx;
                    let yf = y as f64 - cy;
                    let basis = [1.0, xf, yf, xf*xf, yf*yf, xf*yf];
                    for i in 0..6 {
                        atb[i] += basis[i] * v;
                        for j in 0..6 {
                            ata[i][j] += basis[i] * basis[j];
                        }
                    }
                }
            }

            // Solve via Gaussian elimination with partial pivoting
            let coeffs_c = solve6(&ata, &atb);

            // Convert centred coefficients back to origin
            // a0_orig = a0 - a1*cx - a2*cy + a3*cx² + a4*cy² + a5*cx*cy
            // a1_orig = a1 - 2*a3*cx - a5*cy
            // a2_orig = a2 - 2*a4*cy - a5*cx
            // a3_orig = a3, a4_orig = a4, a5_orig = a5
            let a = coeffs_c;
            let orig = [
                a[0] - a[1]*cx - a[2]*cy + a[3]*cx*cx + a[4]*cy*cy + a[5]*cx*cy,
                a[1] - 2.0*a[3]*cx - a[5]*cy,
                a[2] - 2.0*a[4]*cy - a[5]*cx,
                a[3],
                a[4],
                a[5],
            ];

            // Store as f32 — quadratic coefficients are fitting results, not measurements
            for &coeff in &orig {
                payload.extend_from_slice(&(coeff as f32).to_le_bytes());
            }
        }

        EncodedRegion {
            encoder_name: self.name(),
            rate_bits: (payload.len() * 8) as u64, // 72 bytes = 576 bits
            decode_cost: (TILE_SIZE * TILE_SIZE * 3 * 6) as u64,
            encode_cost: (TILE_SIZE * TILE_SIZE * 3 * 36) as u64,
            memory_cost: payload.len() as u64,
            replay_cost: (TILE_SIZE * TILE_SIZE * 3 * 36) as u64,
            payload,
        }
    }

    fn decode(&self, encoded: &EncodedRegion, out: &mut LmsTile) {
        let p = &encoded.payload;
        for c in 0..3 {
            let base = c * 24; // 6 f32 = 24 bytes per channel
            let mut a = [0f32; 6];
            for i in 0..6 {
                a[i] = f32::from_le_bytes(p[base+i*4..base+i*4+4].try_into().unwrap());
            }
            for y in 0..TILE_SIZE {
                for x in 0..TILE_SIZE {
                    let xf = x as f32;
                    let yf = y as f32;
                    let v = a[0]
                        + a[1] * xf
                        + a[2] * yf
                        + a[3] * xf * xf
                        + a[4] * yf * yf
                        + a[5] * xf * yf;
                    out.set(x, y, c, (v * FIX_SCALE as f32).round() as i64);
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

/// Solve 6×6 linear system Ax=b via Gaussian elimination with partial pivoting.
/// Returns zero vector if singular.
fn solve6(a: &[[f64;6];6], b: &[f64;6]) -> [f64;6] {
    let mut m = [[0f64; 7]; 6];
    for i in 0..6 {
        for j in 0..6 { m[i][j] = a[i][j]; }
        m[i][6] = b[i];
    }
    for col in 0..6 {
        // Partial pivot
        let mut max_row = col;
        for row in col+1..6 {
            if m[row][col].abs() > m[max_row][col].abs() { max_row = row; }
        }
        m.swap(col, max_row);
        let pivot = m[col][col];
        if pivot.abs() < 1e-12 { return [0f64; 6]; }
        for row in col+1..6 {
            let factor = m[row][col] / pivot;
            for k in col..7 { m[row][k] -= factor * m[col][k]; }
        }
    }
    // Back substitution
    let mut x = [0f64; 6];
    for i in (0..6).rev() {
        x[i] = m[i][6];
        for j in i+1..6 { x[i] -= m[i][j] * x[j]; }
        x[i] /= m[i][i];
    }
    x
}
