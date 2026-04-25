//! Core types for CIF-RDO v0.

pub const TILE_SIZE: usize = 32;
pub const FIX_SCALE: i64 = 1_000_000;

/// A fixed-size 32×32 LMS tile in fixed-point representation.
/// Values are i64 with scale FIX_SCALE (i.e. 1.0 == 1_000_000).
#[derive(Clone)]
pub struct LmsTile {
    /// Row-major, 32×32×3 fixed-point LMS values.
    pub data: Vec<i64>,
    /// Tile column index in the grid.
    pub grid_x: usize,
    /// Tile row index in the grid.
    pub grid_y: usize,
}

impl LmsTile {
    pub fn new(grid_x: usize, grid_y: usize) -> Self {
        Self {
            data: vec![0i64; TILE_SIZE * TILE_SIZE * 3],
            grid_x,
            grid_y,
        }
    }

    /// Get pixel channel value at (x, y, channel).
    #[inline]
    pub fn get(&self, x: usize, y: usize, c: usize) -> i64 {
        self.data[(y * TILE_SIZE + x) * 3 + c]
    }

    /// Set pixel channel value at (x, y, channel).
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, c: usize, v: i64) {
        self.data[(y * TILE_SIZE + x) * 3 + c] = v;
    }
}

/// The output of encoding a tile with one candidate encoder.
#[derive(Clone)]
pub struct EncodedRegion {
    /// Encoder that produced this region.
    pub encoder_name: &'static str,
    /// Raw payload bytes.
    pub payload: Vec<u8>,
    /// Precomputed rate in bits (= payload.len() * 8 for most encoders).
    pub rate_bits: u64,
    /// Abstract decode operation count (for future ζ term).
    pub decode_cost: u64,
    /// Abstract encode operation count (for future δ term).
    pub encode_cost: u64,
    /// Abstract memory cost in bytes (for future ε term).
    pub memory_cost: u64,
    /// Abstract replay operation count (for future ζ term).
    pub replay_cost: u64,
}

/// The interface every v0 encoder must implement.
pub trait RegionEncoder: Send + Sync {
    fn name(&self) -> &'static str;
    fn encode(&self, tile: &LmsTile) -> EncodedRegion;
    fn decode(&self, encoded: &EncodedRegion, out: &mut LmsTile);
    fn rate_bits(&self, encoded: &EncodedRegion) -> u64;
    fn decode_cost(&self, encoded: &EncodedRegion) -> u64;
    fn encode_cost(&self, encoded: &EncodedRegion) -> u64;
    fn memory_cost(&self, encoded: &EncodedRegion) -> u64;
    fn replay_cost(&self, encoded: &EncodedRegion) -> u64;
}

/// A selected region entry for tree.json.
#[derive(Clone)]
pub struct RegionEntry {
    pub id: usize,
    pub grid_x: usize,
    pub grid_y: usize,
    pub encoder: &'static str,
    pub payload_offset: u64,
    pub payload_len: u64,
    pub payload_digest: String,
    pub rate_bits: u64,
    pub d_oklab_fixed: i64,
    pub j_fixed: i64,
    pub decode_cost: u64,
    pub encode_cost: u64,
    pub memory_cost: u64,
    pub replay_cost: u64,
}

/// v0 objective weights. Only α and β are active.
pub struct Weights {
    pub alpha: i64,         // rate weight × FIX_SCALE
    pub quality_lambda: i64, // β = quality_lambda (fixed-point)
}

impl Weights {
    pub fn v0(quality_lambda: i64) -> Self {
        Self {
            alpha: FIX_SCALE,  // α = 1.0
            quality_lambda,
        }
    }
}
