//! micro_siren_tile encoder.
//!
//! A minimal SIREN (1 hidden layer, 8 wide) fitted to each 32x32 tile.
//! Architecture: input=2, hidden=[8], output=3, activation=sin, omega=1.0
//! Parameters: W0(8x2)+b0(8) + Wout(3x8)+bout(3) = 51 f32 = 204 bytes
//! rate_bits: 1632

use crate::rdo::types::{EncodedRegion, LmsTile, RegionEncoder, TILE_SIZE, FIX_SCALE};

const HW: usize = 8;       // hidden width
const INP: usize = 2;      // input dims (x, y)
const OUT: usize = 3;      // output dims (L, M, S)
const OMEGA: f32 = 1.0;
const LR: f32 = 0.01;
const STEPS: usize = 64;
const BATCH: usize = 16;
const N_PARAMS: usize = HW*INP + HW + OUT*HW + OUT; // 51

pub struct MicroSirenTile;

/// Deterministic LCG for pixel sampling — seeded from tile position.
struct TileRng { state: u64 }
impl TileRng {
    fn new(grid_x: usize, grid_y: usize) -> Self {
        let s = (grid_x as u64).wrapping_mul(2654435761)
            ^ (grid_y as u64).wrapping_mul(2246822519);
        Self { state: s | 1 }
    }
    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        (self.next() as f64 / u64::MAX as f64) as f32
    }
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next() as usize) % n
    }
}

/// Forward pass: (x,y) -> [L,M,S] in [0,1] range.
fn forward(x: f32, y: f32, w: &[f32]) -> [f32; OUT] {
    // Layer 0: z0 = W0@[x,y] + b0, h0 = sin(omega*z0)
    let mut h = [0f32; HW];
    for r in 0..HW {
        let mut s = w[HW*INP + r]; // bias
        s += w[r*INP + 0] * x;
        s += w[r*INP + 1] * y;
        h[r] = (OMEGA * s).sin();
    }
    // Output layer: o = Wout@h + bout (linear)
    let off = HW*INP + HW;
    let mut o = [0f32; OUT];
    for r in 0..OUT {
        let mut s = w[off + OUT*HW + r]; // bias
        for c in 0..HW { s += w[off + r*HW + c] * h[c]; }
        o[r] = s;
    }
    o
}

/// Train the micro SIREN on a tile using mini-batch SGD.
fn train(tile: &LmsTile) -> [f32; N_PARAMS] {
    let mut rng = TileRng::new(tile.grid_x, tile.grid_y);
    let tw = TILE_SIZE as f32;

    // Init weights small random
    let mut w = [0f32; N_PARAMS];
    for v in w.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * 0.1;
    }

    let n_px = TILE_SIZE * TILE_SIZE;

    for _step in 0..STEPS {
        let mut grads = [0f32; N_PARAMS];

        for _b in 0..BATCH {
            let idx = rng.next_usize(n_px);
            let px = idx % TILE_SIZE;
            let py = idx / TILE_SIZE;
            let nx = (px as f32 + 0.5) / tw;
            let ny = (py as f32 + 0.5) / tw;

            // Target in [0,1]
            let target = [
                tile.get(px, py, 0) as f32 / FIX_SCALE as f32,
                tile.get(px, py, 1) as f32 / FIX_SCALE as f32,
                tile.get(px, py, 2) as f32 / FIX_SCALE as f32,
            ];

            // Forward
            let mut h = [0f32; HW];
            let mut z = [0f32; HW];
            for r in 0..HW {
                let mut s = w[HW*INP + r];
                s += w[r*INP + 0] * nx;
                s += w[r*INP + 1] * ny;
                z[r] = OMEGA * s;
                h[r] = z[r].sin();
            }
            let off = HW*INP + HW;
            let mut o = [0f32; OUT];
            for r in 0..OUT {
                let mut s = w[off + OUT*HW + r];
                for c in 0..HW { s += w[off + r*HW + c] * h[c]; }
                o[r] = s;
            }

            // Loss gradient dL/do = 2/OUT * (o - target)
            let mut d_o = [0f32; OUT];
            for c in 0..OUT { d_o[c] = (2.0 / OUT as f32) * (o[c] - target[c]); }

            // Output layer grads
            for r in 0..OUT {
                grads[off + OUT*HW + r] += d_o[r];
                for c in 0..HW { grads[off + r*HW + c] += d_o[r] * h[c]; }
            }

            // d_h = Wout^T @ d_o
            let mut d_h = [0f32; HW];
            for c in 0..HW {
                for r in 0..OUT { d_h[c] += w[off + r*HW + c] * d_o[r]; }
            }

            // Layer 0 grads: d_z = d_h * omega * cos(z)
            for j in 0..HW {
                let d_z = d_h[j] * OMEGA * z[j].cos();
                grads[HW*INP + j] += d_z; // d_b0
                grads[j*INP + 0] += d_z * nx;
                grads[j*INP + 1] += d_z * ny;
            }
        }

        // SGD update with clipping
        let scale = LR / BATCH as f32;
        for i in 0..N_PARAMS {
            w[i] -= (grads[i] * scale).clamp(-0.1, 0.1);
        }
    }
    w
}

impl RegionEncoder for MicroSirenTile {
    fn name(&self) -> &'static str { "micro_siren_tile" }

    fn encode(&self, tile: &LmsTile) -> EncodedRegion {
        let w = train(tile);
        let mut payload = Vec::with_capacity(N_PARAMS * 4);
        for &v in &w { payload.extend_from_slice(&v.to_le_bytes()); }

        EncodedRegion {
            encoder_name: self.name(),
            rate_bits: (payload.len() * 8) as u64,
            decode_cost: (TILE_SIZE * TILE_SIZE * (HW*INP + HW + OUT*HW + OUT)) as u64,
            encode_cost: (STEPS * BATCH * (HW*INP + HW + OUT*HW + OUT)) as u64,
            memory_cost: payload.len() as u64,
            replay_cost: (STEPS * BATCH * (HW*INP + HW + OUT*HW + OUT)) as u64,
            payload,
        }
    }

    fn decode(&self, encoded: &EncodedRegion, out: &mut LmsTile) {
        let p = &encoded.payload;
        let mut w = [0f32; N_PARAMS];
        for i in 0..N_PARAMS {
            w[i] = f32::from_le_bytes(p[i*4..i*4+4].try_into().unwrap());
        }
        let tw = TILE_SIZE as f32;
        for y in 0..TILE_SIZE {
            for x in 0..TILE_SIZE {
                let nx = (x as f32 + 0.5) / tw;
                let ny = (y as f32 + 0.5) / tw;
                let o = forward(nx, ny, &w);
                out.set(x, y, 0, (o[0].clamp(0.0,1.0) * FIX_SCALE as f32) as i64);
                out.set(x, y, 1, (o[1].clamp(0.0,1.0) * FIX_SCALE as f32) as i64);
                out.set(x, y, 2, (o[2].clamp(0.0,1.0) * FIX_SCALE as f32) as i64);
            }
        }
    }

    fn rate_bits(&self, e: &EncodedRegion) -> u64 { e.rate_bits }
    fn decode_cost(&self, e: &EncodedRegion) -> u64 { e.decode_cost }
    fn encode_cost(&self, e: &EncodedRegion) -> u64 { e.encode_cost }
    fn memory_cost(&self, e: &EncodedRegion) -> u64 { e.memory_cost }
    fn replay_cost(&self, e: &EncodedRegion) -> u64 { e.replay_cost }
}
