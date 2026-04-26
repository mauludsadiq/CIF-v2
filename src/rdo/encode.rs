//! CIF-RDO v0 encoder — top-level encode function.

use anyhow::Result;
use std::path::Path;
use std::fs;
use sha2::{Digest, Sha256};
use serde_json::json;

use crate::rdo::types::{LmsTile, RegionEncoder, Weights, FIX_SCALE, TILE_SIZE};
use crate::rdo::encoders::constant::ConstantLms;
use crate::rdo::encoders::affine::AffineLms;
use crate::rdo::encoders::quadratic::QuadraticLms;
use crate::rdo::encoders::wavelet::WaveletTile;
use crate::rdo::encoders::edge::EdgeTile;
use crate::rdo::encoders::siren::MicroSirenTile;
use crate::rdo::select::{select_encoder, Selection};

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new(); h.update(bytes);
    let r = h.finalize();
    const HEX: &[u8;16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for b in r.iter() { s.push(HEX[(b>>4) as usize] as char); s.push(HEX[(b&15) as usize] as char); }
    format!("sha256:{s}")
}

/// Convert f32 LMS pixel to fixed-point i64 (scale FIX_SCALE).
#[inline]
fn lms_f32_to_fixed(v: f32) -> i64 {
    (v * FIX_SCALE as f32).round() as i64
}

pub fn rdo_encode(input: &Path, out: &Path, tile_size: u32, quality: f64) -> Result<()> {
    // Re-use CIF v2 canonicalize path — read image, convert to LMS f32 tensor
    let img = image::open(input)
        .with_context(|| format!("decode {}", input.display()))?
        .to_rgba8();
    let (iw, ih) = img.dimensions();

    // Pad to tile grid
    let tw = tile_size as usize;
    let grid_w = (iw as usize + tw - 1) / tw;
    let grid_h = (ih as usize + tw - 1) / tw;
    let pad_w = grid_w * tw;
    let pad_h = grid_h * tw;

    // Build padded LMS fixed-point buffer
    let mut lms_buf = vec![[0i64; 3]; pad_w * pad_h];
    for y in 0..ih as usize {
        for x in 0..iw as usize {
            let p = img.get_pixel(x as u32, y as u32);
            let r = srgb_to_linear(p[0]);
            let g = srgb_to_linear(p[1]);
            let b = srgb_to_linear(p[2]);
            let l = 0.31399*r + 0.63951*g + 0.04649*b;
            let m = 0.15537*r + 0.75789*g + 0.08670*b;
            let s = 0.01775*r + 0.10944*g + 0.87257*b;
            lms_buf[y * pad_w + x] = [
                lms_f32_to_fixed(l),
                lms_f32_to_fixed(m),
                lms_f32_to_fixed(s),
            ];
        }
    }
    // Edge-extend padding
    for y in 0..pad_h {
        for x in 0..pad_w {
            if y >= ih as usize || x >= iw as usize {
                let sy = y.min(ih as usize - 1);
                let sx = x.min(iw as usize - 1);
                lms_buf[y * pad_w + x] = lms_buf[sy * pad_w + sx];
            }
        }
    }

    // Build candidate encoder set
    let encoders: Vec<Box<dyn RegionEncoder>> = vec![
        Box::new(ConstantLms),
        Box::new(AffineLms),
        Box::new(QuadraticLms),
        Box::new(WaveletTile),
        Box::new(EdgeTile),
        Box::new(MicroSirenTile),
    ];

    let quality_lambda = (quality * FIX_SCALE as f64).round() as i64;
    let weights = Weights::v0(quality_lambda);

    // Process tiles
    let mut regions = Vec::new();
    let mut payloads_bin: Vec<u8> = Vec::new();
    let mut payload_offset: u64 = 0;

    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let mut tile = LmsTile::new(gx, gy);
            // Fill tile from padded buffer
            for ty in 0..tw {
                for tx in 0..tw {
                    let px = gx * tw + tx;
                    let py = gy * tw + ty;
                    let lms = lms_buf[py * pad_w + px];
                    tile.set(tx, ty, 0, lms[0]);
                    tile.set(tx, ty, 1, lms[1]);
                    tile.set(tx, ty, 2, lms[2]);
                }
            }

            let sel = select_encoder(&tile, &encoders, &weights);
            let id = gy * grid_w + gx;
            let plen = sel.encoded.payload.len() as u64;

            regions.push(json!({
                "id": id,
                "bounds": [gx*tw, gy*tw, (gx+1)*tw, (gy+1)*tw],
                "encoder": sel.encoded.encoder_name,
                "payload_offset": payload_offset,
                "payload_len": plen,
                "payload_digest": sel.payload_digest,
                "rate_bits": sel.encoded.rate_bits,
                "d_oklab_fixed": sel.d_oklab_fixed,
                "j_fixed": sel.j_fixed,
                "decode_cost": sel.encoded.decode_cost,
                "encode_cost": sel.encoded.encode_cost,
                "memory_cost": sel.encoded.memory_cost,
                "replay_cost": sel.encoded.replay_cost,
            }));

            payloads_bin.extend_from_slice(&sel.encoded.payload);
            payload_offset += plen;
        }
    }

    // Write artifact
    fs::create_dir_all(out)?;
    fs::create_dir_all(out.join("regions"))?;

    let tree = json!({
        "version": "CIF-RDO-v0",
        "tile_size": tw,
        "grid_width": grid_w,
        "grid_height": grid_h,
        "visible_width": iw,
        "visible_height": ih,
        "quality_lambda": quality_lambda.to_string(),
        "regions": regions,
    });

    let tree_bytes = serde_json::to_string_pretty(&tree)? + "\n";
    fs::write(out.join("regions/tree.json"), &tree_bytes)?;
    fs::write(out.join("regions/payloads.bin"), &payloads_bin)?;

    let h_tree = sha256_hex(tree_bytes.as_bytes());
    let h_payloads = sha256_hex(&payloads_bin);

    let manifest = json!({
        "format": "CIF-RDO-v0",
        "operator": "Phi-CIF-RDO-v0",
        "visible_width": iw,
        "visible_height": ih,
        "tile_size": tw,
        "quality_lambda": quality_lambda.to_string(),
        "hashes": {
            "tree": h_tree,
            "payloads": h_payloads,
        }
    });
    let manifest_bytes = serde_json::to_string_pretty(&manifest)? + "\n";
    fs::write(out.join("manifest.json"), &manifest_bytes)?;
    let h_manifest = sha256_hex(manifest_bytes.as_bytes());

    // Artifact digest
    let mut ad_buf = Vec::new();
    ad_buf.extend_from_slice(h_tree.as_bytes());
    ad_buf.extend_from_slice(h_payloads.as_bytes());
    ad_buf.extend_from_slice(h_manifest.as_bytes());
    let artifact_digest = sha256_hex(&ad_buf);

    let receipt = json!({
        "receipt_version": "FARD-CIF-RDO-RECEIPT-1",
        "operator": "Phi-CIF-RDO-v0",
        "artifact_digest": artifact_digest,
        "tree_digest": h_tree,
        "payloads_digest": h_payloads,
        "manifest_digest": h_manifest,
    });
    fs::write(out.join("receipt.json"), serde_json::to_string_pretty(&receipt)? + "\n")?;

    println!("{}", serde_json::to_string_pretty(&json!({
        "ok": true,
        "artifact_digest": artifact_digest,
        "grid": format!("{}x{}", grid_w, grid_h),
        "tiles": grid_w * grid_h,
        "out": out.to_string_lossy(),
    }))?);

    Ok(())
}

#[inline]
fn srgb_to_linear(v: u8) -> f32 {
    (v as f32 / 255.0).powf(2.2)
}

// bring in context trait
use anyhow::Context;
use image::GenericImageView;
