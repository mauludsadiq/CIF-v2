//! CIF-RDO v0 deterministic tile selector.
//!
//! For each tile, evaluates all candidate encoders and selects the one
//! minimizing J_fixed. Ties broken by the spec-defined 4-rule order.

use crate::rdo::types::{EncodedRegion, LmsTile, RegionEncoder, Weights};
use crate::rdo::objective::{d_oklab_fixed, j_fixed};
use sha2::{Digest, Sha256};

/// Result of selecting the best encoder for a tile.
pub struct Selection {
    pub encoded: EncodedRegion,
    pub d_oklab_fixed: i64,
    pub j_fixed: i64,
    pub payload_digest: String,
}

/// Select the best encoder for a tile from the candidate set.
/// Tie-breaking (spec order):
///   1. lowest j_fixed
///   2. lowest rate_bits
///   3. lexicographically smallest encoder name
///   4. lexicographically smallest SHA-256 digest of payload
pub fn select_encoder(
    tile: &LmsTile,
    encoders: &[Box<dyn RegionEncoder>],
    weights: &Weights,
) -> Selection {
    let mut best: Option<(Selection, u64, &'static str, String)> = None;

    for encoder in encoders {
        let encoded = encoder.encode(tile);

        // Decode back to measure distortion
        let mut decoded = LmsTile::new(tile.grid_x, tile.grid_y);
        encoder.decode(&encoded, &mut decoded);

        let d = d_oklab_fixed(tile, &decoded);
        let j = j_fixed(encoded.rate_bits, d, weights.quality_lambda);

        // Compute payload digest for tie-breaking rule 4
        let digest = {
            let mut h = Sha256::new();
            h.update(&encoded.payload);
            format!("sha256:{}", hex_lower(&h.finalize()))
        };

        let candidate = Selection {
            payload_digest: digest.clone(),
            d_oklab_fixed: d,
            j_fixed: j,
            encoded,
        };

        let is_better = match &best {
            None => true,
            Some((prev, prev_rate, prev_name, prev_digest)) => {
                // Rule 1: lowest j_fixed
                if j < prev.j_fixed { true }
                else if j > prev.j_fixed { false }
                // Rule 2: lowest rate_bits
                else if candidate.encoded.rate_bits < *prev_rate { true }
                else if candidate.encoded.rate_bits > *prev_rate { false }
                // Rule 3: lexicographically smallest encoder name
                else if candidate.encoded.encoder_name < *prev_name { true }
                else if candidate.encoded.encoder_name > *prev_name { false }
                // Rule 4: lexicographically smallest payload digest
                else { digest < *prev_digest }
            }
        };

        if is_better {
            let rate = candidate.encoded.rate_bits;
            let name = candidate.encoded.encoder_name;
            let dig = digest.clone();
            best = Some((candidate, rate, name, dig));
        }
    }

    best.map(|(s, _, _, _)| s).unwrap()
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 15) as usize] as char);
    }
    s
}
