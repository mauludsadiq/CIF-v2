//! CIF-RDO v0 renderer.
//!
//! Decodes each tile using the selected encoder and composites
//! the result into an output PNG at any target resolution.

use anyhow::Result;
use std::path::Path;
use std::fs;
use image::{ImageBuffer, Rgba};
use serde_json::Value;

use crate::rdo::types::{LmsTile, TILE_SIZE, FIX_SCALE};
use crate::rdo::encoders::constant::ConstantLms;
use crate::rdo::encoders::affine::AffineLms;
use crate::rdo::encoders::quadratic::QuadraticLms;
use crate::rdo::types::RegionEncoder;

fn lms_to_srgb(l: i64, m: i64, s: i64) -> [u8; 3] {
    let lf = l as f32 / FIX_SCALE as f32;
    let mf = m as f32 / FIX_SCALE as f32;
    let sf = s as f32 / FIX_SCALE as f32;
    let r = ( 5.47221206*lf - 4.64196010*mf + 0.16963708*sf).clamp(0.0,1.0);
    let g = (-1.12524190*lf + 2.29317094*mf - 0.16789520*sf).clamp(0.0,1.0);
    let b = ( 0.02980165*lf - 0.19318073*mf + 1.16364789*sf).clamp(0.0,1.0);
    let f = |v: f32| (v.powf(1.0/2.2)*255.0).round() as u8;
    [f(r), f(g), f(b)]
}

fn get_encoder(name: &str) -> Box<dyn RegionEncoder> {
    match name {
        "constant_lms"  => Box::new(ConstantLms),
        "affine_lms"    => Box::new(AffineLms),
        "quadratic_lms" => Box::new(QuadraticLms),
        other => panic!("unknown encoder: {other}"),
    }
}

pub fn rdo_render(artifact: &Path, out: &Path, width: u32, height: u32) -> Result<()> {
    let tree_bytes    = fs::read(artifact.join("regions/tree.json"))?;
    let payload_bytes = fs::read(artifact.join("regions/payloads.bin"))?;
    let receipt: Value = serde_json::from_slice(&fs::read(artifact.join("receipt.json"))?)?;
    let tree: Value   = serde_json::from_slice(&tree_bytes)?;

    let artifact_digest = receipt["artifact_digest"].as_str().unwrap_or("unknown").to_string();
    let visible_w = tree["visible_width"].as_u64().unwrap() as usize;
    let visible_h = tree["visible_height"].as_u64().unwrap() as usize;
    let tile_size = tree["tile_size"].as_u64().unwrap() as usize;
    let grid_w = tree["grid_width"].as_u64().unwrap() as usize;
    let grid_h = tree["grid_height"].as_u64().unwrap() as usize;

    // Decode all tiles into a full LMS buffer
    let pad_w = grid_w * tile_size;
    let pad_h = grid_h * tile_size;
    let mut lms_buf = vec![[0i64; 3]; pad_w * pad_h];

    let regions = tree["regions"].as_array().unwrap();
    for region in regions {
        let encoder_name = region["encoder"].as_str().unwrap();
        let offset = region["payload_offset"].as_u64().unwrap() as usize;
        let len    = region["payload_len"].as_u64().unwrap() as usize;
        let gx     = region["bounds"][0].as_u64().unwrap() as usize / tile_size;
        let gy     = region["bounds"][1].as_u64().unwrap() as usize / tile_size;

        let encoder = get_encoder(encoder_name);
        let encoded = crate::rdo::types::EncodedRegion {
            encoder_name: encoder.name(),
            payload: payload_bytes[offset..offset+len].to_vec(),
            rate_bits: region["rate_bits"].as_u64().unwrap_or(0),
            decode_cost: region["decode_cost"].as_u64().unwrap_or(0),
            encode_cost: region["encode_cost"].as_u64().unwrap_or(0),
            memory_cost: region["memory_cost"].as_u64().unwrap_or(0),
            replay_cost: region["replay_cost"].as_u64().unwrap_or(0),
        };

        let mut tile = LmsTile::new(gx, gy);
        encoder.decode(&encoded, &mut tile);

        // Write tile into lms_buf
        for ty in 0..tile_size {
            for tx in 0..tile_size {
                let px = gx * tile_size + tx;
                let py = gy * tile_size + ty;
                lms_buf[py * pad_w + px] = [
                    tile.get(tx, ty, 0),
                    tile.get(tx, ty, 1),
                    tile.get(tx, ty, 2),
                ];
            }
        }
    }

    // Render to target resolution via bilinear sampling
    let cw = visible_w as f32;
    let ch = visible_h as f32;
    let ow = width as f32;
    let oh = height as f32;

    let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);

    for py in 0..height as usize {
        for px in 0..width as usize {
            let cx = (px as f32 + 0.5) * cw / ow - 0.5;
            let cy = (py as f32 + 0.5) * ch / oh - 0.5;

            let x0 = (cx.floor() as isize).clamp(0, visible_w as isize - 1) as usize;
            let y0 = (cy.floor() as isize).clamp(0, visible_h as isize - 1) as usize;
            let x1 = (x0 + 1).min(visible_w - 1);
            let y1 = (y0 + 1).min(visible_h - 1);
            let tx = (cx - cx.floor()).clamp(0.0, 1.0);
            let ty = (cy - cy.floor()).clamp(0.0, 1.0);

            let s = |xi: usize, yi: usize| lms_buf[yi * pad_w + xi];
            let lerp = |a: [i64;3], b: [i64;3], t: f32| -> [i64;3] {[
                (a[0] as f32 + (b[0]-a[0]) as f32 * t) as i64,
                (a[1] as f32 + (b[1]-a[1]) as f32 * t) as i64,
                (a[2] as f32 + (b[2]-a[2]) as f32 * t) as i64,
            ]};

            let top    = lerp(s(x0,y0), s(x1,y0), tx);
            let bottom = lerp(s(x0,y1), s(x1,y1), tx);
            let lms    = lerp(top, bottom, ty);

            let rgb = lms_to_srgb(lms[0], lms[1], lms[2]);
            img.put_pixel(px as u32, py as u32, Rgba([rgb[0], rgb[1], rgb[2], 255]));
        }
    }

    img.save(out)?;

    let render_digest = {
        use sha2::{Digest as _, Sha256};
        let bytes = fs::read(out)?;
        let mut h = Sha256::new(); h.update(&bytes);
        let r = h.finalize();
        const HEX: &[u8;16] = b"0123456789abcdef";
        let mut s = String::with_capacity(64);
        for b in r.iter() { s.push(HEX[(b>>4) as usize] as char); s.push(HEX[(b&15) as usize] as char); }
        format!("sha256:{s}")
    };

    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "ok": true,
        "artifact_digest": artifact_digest,
        "projection": {
            "width": width,
            "height": height,
            "out": out.to_string_lossy(),
            "render_digest": render_digest,
        }
    }))?);

    Ok(())
}
