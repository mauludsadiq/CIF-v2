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
    let tree_bytes     = fs::read(artifact.join("regions/tree.json"))?;
    let payloads_bytes = fs::read(artifact.join("regions/payloads.bin"))?;

    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    let receipt:  Value = serde_json::from_slice(&receipt_bytes)?;
    let tree:     Value = serde_json::from_slice(&tree_bytes)?;

    // Step 1: verify component digests against manifest
    let h_tree     = sha256_hex(&tree_bytes);
    let h_payloads = sha256_hex(&payloads_bytes);

    let m_tree = manifest["hashes"]["tree"].as_str()
        .ok_or_else(|| anyhow::anyhow!("missing manifest.hashes.tree"))?;
    let m_payloads = manifest["hashes"]["payloads"].as_str()
        .ok_or_else(|| anyhow::anyhow!("missing manifest.hashes.payloads"))?;

    if h_tree != m_tree {
        bail!("tree.json digest mismatch: got {h_tree}, expected {m_tree}");
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

    // Step 4: verify each region's payload_digest
    let regions = tree["regions"].as_array()
        .ok_or_else(|| anyhow::anyhow!("missing regions array"))?;

    let mut regions_verified = 0usize;
    for region in regions {
        let offset = region["payload_offset"].as_u64()
            .ok_or_else(|| anyhow::anyhow!("missing payload_offset"))? as usize;
        let len = region["payload_len"].as_u64()
            .ok_or_else(|| anyhow::anyhow!("missing payload_len"))? as usize;
        let expected_digest = region["payload_digest"].as_str()
            .ok_or_else(|| anyhow::anyhow!("missing payload_digest"))?;
        let id = region["id"].as_u64().unwrap_or(0);

        if offset + len > payloads_bytes.len() {
            bail!("region {id}: payload out of bounds");
        }
        let actual_digest = sha256_hex(&payloads_bytes[offset..offset+len]);
        if actual_digest != expected_digest {
            bail!("region {id}: payload digest mismatch");
        }
        regions_verified += 1;
    }

    Ok(serde_json::json!({
        "ok": true,
        "artifact_digest": artifact_digest,
        "regions_verified": regions_verified,
        "grid": format!("{}x{}",
            tree["grid_width"].as_u64().unwrap_or(0),
            tree["grid_height"].as_u64().unwrap_or(0)),
    }))
}
