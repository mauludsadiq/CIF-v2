use crate::rdo::types::{EncodedRegion, LmsTile, RegionEncoder, TILE_SIZE};

pub struct ConstantLms;

impl RegionEncoder for ConstantLms {
    fn name(&self) -> &'static str { "constant_lms" }

    fn encode(&self, tile: &LmsTile) -> EncodedRegion {
        let n = (TILE_SIZE * TILE_SIZE) as i64;
        let mut sum = [0i64; 3];
        for y in 0..TILE_SIZE {
            for x in 0..TILE_SIZE {
                for c in 0..3 { sum[c] += tile.get(x, y, c); }
            }
        }
        let mean = [sum[0]/n, sum[1]/n, sum[2]/n];
        let mut payload = Vec::with_capacity(12);
        for c in 0..3 { payload.extend_from_slice(&(mean[c] as i32).to_le_bytes()); }
        EncodedRegion {
            encoder_name: self.name(),
            payload,
            rate_bits: 96,
            decode_cost: (TILE_SIZE * TILE_SIZE * 3) as u64,
            encode_cost: (TILE_SIZE * TILE_SIZE * 3) as u64,
            memory_cost: 12,
            replay_cost: (TILE_SIZE * TILE_SIZE * 3) as u64,
        }
    }

    fn decode(&self, encoded: &EncodedRegion, out: &mut LmsTile) {
        let p = &encoded.payload;
        let mean = [
            i32::from_le_bytes(p[0..4].try_into().unwrap()) as i64,
            i32::from_le_bytes(p[4..8].try_into().unwrap()) as i64,
            i32::from_le_bytes(p[8..12].try_into().unwrap()) as i64,
        ];
        for y in 0..TILE_SIZE {
            for x in 0..TILE_SIZE {
                for c in 0..3 { out.set(x, y, c, mean[c]); }
            }
        }
    }

    fn rate_bits(&self, e: &EncodedRegion) -> u64 { e.rate_bits }
    fn decode_cost(&self, e: &EncodedRegion) -> u64 { e.decode_cost }
    fn encode_cost(&self, e: &EncodedRegion) -> u64 { e.encode_cost }
    fn memory_cost(&self, e: &EncodedRegion) -> u64 { e.memory_cost }
    fn replay_cost(&self, e: &EncodedRegion) -> u64 { e.replay_cost }
}
