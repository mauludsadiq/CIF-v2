//! CIF-RDO v0 artifact verifier.

use anyhow::{bail, Result};
use std::path::Path;
use std::fs;
use sha2::{Digest, Sha256};
use serde_json::Value;

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new(); h.update(bytes);
    let r = h.finalize();
    const HEX: &[u8;16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for b in r.iter() {
        s.push(HEX[(b>>4) as usize] as char);
        s.push(HEX[(b&15) as usize] as char);
    }
    format!("sha256:{s}")
}

pub fn rdo_verify(artifact: &Path) -> Result<serde_json::Value> {
    let manifest_bytes = fs::read(artifact.join("manifest.json"))?;
    let receipt_bytes  = fs::read(artifact.join("receipt.json"))?;
    let tree_bytes     = fs::read(artifact.join("regions/tree.bin"))?;
    let payloads_bytes = fs::read(artifact.join("regions/payloads.bin"))?;

    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    let receipt:  Value = serde_json::from_slice(&receipt_bytes)?;

    // Step 1: verify component digests against manifest
    let h_tree     = sha256_hex(&tree_bytes);
    let h_payloads = sha256_hex(&payloads_bytes);

    let m_tree = manifest["hashes"]["tree_bin"].as_str()
        .ok_or_else(|| anyhow::anyhow!("missing manifest.hashes.tree_bin"))?;
    let m_payloads = manifest["hashes"]["payloads"].as_str()
        .ok_or_else(|| anyhow::anyhow!("missing manifest.hashes.payloads"))?;

    if h_tree != m_tree {
        bail!("tree.bin digest mismatch: got {h_tree}, expected {m_tree}");
    }
    if h_payloads != m_payloads {
        bail!("payloads.bin digest mismatch: got {h_payloads}, expected {m_payloads}");
    }

    // Step 2: verify manifest digest against receipt
    let h_manifest = sha256_hex(&manifest_bytes);
    let r_manifest = receipt["manifest_digest"].as_str()
        .ok_or_else(|| anyhow::anyhow!("missing receipt.manifest_digest"))?;
    if h_manifest != r_manifest {
        bail!("manifest digest mismatch: got {h_manifest}, expected {r_manifest}");
    }

    // Step 3: recompute artifact digest
    let mut ad_buf = Vec::new();
    ad_buf.extend_from_slice(h_tree.as_bytes());
    ad_buf.extend_from_slice(h_payloads.as_bytes());
    ad_buf.extend_from_slice(h_manifest.as_bytes());
    let artifact_digest = sha256_hex(&ad_buf);

    let r_artifact = receipt["artifact_digest"].as_str()
        .ok_or_else(|| anyhow::anyhow!("missing receipt.artifact_digest"))?;
    if artifact_digest != r_artifact {
        bail!("artifact digest mismatch: got {artifact_digest}, expected {r_artifact}");
    }

    // tree.bin stores no per-region digests — payload integrity is covered
    // by the payloads digest in the manifest.
    let (hdr, tile_records) = crate::rdo::tree_bin::read_tree(&tree_bytes)?;
    let n_tiles = tile_records.len();
    // Verify payloads.bin length matches sum of payload_lens
    let expected_len: usize = tile_records.iter().map(|t| t.payload_len as usize).sum();
    if expected_len != payloads_bytes.len() {
        bail!("payloads.bin length mismatch: tree.bin expects {expected_len}, got {}", payloads_bytes.len());
    }

    Ok(serde_json::json!({
        "ok": true,
        "artifact_digest": artifact_digest,
        "tiles_verified": n_tiles,
        "grid": format!("{}x{}", hdr.grid_w, hdr.grid_h),
    }))
}
