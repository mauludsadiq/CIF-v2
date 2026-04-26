//! External codec comparison for rdo-bench.
//!
//! Calls avifenc/avifdec, cjxl/djxl, cwebp/dwebp and measures
//! D_oklab against the canonical LMS reference.

use anyhow::Result;
use std::path::Path;
use std::process::Command;
use std::time::Instant;
use image::GenericImageView;

use crate::rdo::types::{LmsTile, TILE_SIZE, FIX_SCALE};
use crate::rdo::objective::d_oklab_fixed;

fn srgb_to_linear(v: u8) -> f32 { (v as f32 / 255.0).powf(2.2) }

fn rgb_to_lms(r: f32, g: f32, b: f32) -> [f32; 3] {
    [
        0.31399*r + 0.63951*g + 0.04649*b,
        0.15537*r + 0.75789*g + 0.08670*b,
        0.01775*r + 0.10944*g + 0.87257*b,
    ]
}

/// Load PNG as LMS fixed-point buffer (width, height, data).
pub fn load_lms(path: &Path) -> Result<(usize, usize, Vec<[i64;3]>)> {
    let img = image::open(path)?.to_rgba8();
    let (w, h) = img.dimensions();
    let mut data = Vec::with_capacity((w*h) as usize);
    for p in img.pixels() {
        let r = srgb_to_linear(p[0]);
        let g = srgb_to_linear(p[1]);
        let b = srgb_to_linear(p[2]);
        let lms = rgb_to_lms(r, g, b);
        data.push([
            (lms[0] * FIX_SCALE as f32).round() as i64,
            (lms[1] * FIX_SCALE as f32).round() as i64,
            (lms[2] * FIX_SCALE as f32).round() as i64,
        ]);
    }
    Ok((w as usize, h as usize, data))
}

/// Compute mean D_oklab between two LMS buffers tiled into 32x32 tiles.
pub fn mean_d_oklab(w: usize, h: usize, ref_lms: &[[i64;3]], dec_lms: &[[i64;3]]) -> f64 {
    let tw = TILE_SIZE;
    let grid_w = (w + tw - 1) / tw;
    let grid_h = (h + tw - 1) / tw;
    let mut total = 0i64;
    let mut count = 0usize;

    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let mut ref_tile = LmsTile::new(gx, gy);
            let mut dec_tile = LmsTile::new(gx, gy);
            for ty in 0..tw {
                for tx in 0..tw {
                    let px = (gx*tw+tx).min(w-1);
                    let py = (gy*tw+ty).min(h-1);
                    let idx = py*w+px;
                    ref_tile.set(tx, ty, 0, ref_lms[idx][0]);
                    ref_tile.set(tx, ty, 1, ref_lms[idx][1]);
                    ref_tile.set(tx, ty, 2, ref_lms[idx][2]);
                    dec_tile.set(tx, ty, 0, dec_lms[idx][0]);
                    dec_tile.set(tx, ty, 1, dec_lms[idx][1]);
                    dec_tile.set(tx, ty, 2, dec_lms[idx][2]);
                }
            }
            total += d_oklab_fixed(&ref_tile, &dec_tile);
            count += 1;
        }
    }
    total as f64 / count as f64 / FIX_SCALE as f64
}

pub struct CodecResult {
    pub codec: &'static str,
    pub size_bytes: u64,
    pub encode_ms: u64,
    pub decode_ms: u64,
    pub d_oklab: f64,
}

/// Run AVIF encode/decode and measure.
pub fn bench_avif(input: &Path, quality: u32) -> Result<CodecResult> {
    let enc_out = std::env::temp_dir().join("cifv2_cmp.avif");
    let dec_out = std::env::temp_dir().join("cifv2_cmp_avif.png");

    let t0 = Instant::now();
    Command::new("avifenc")
        .args(["--qcolor", &quality.to_string(), "--speed", "6"])
        .arg(input).arg(&enc_out)
        .output()?;
    let encode_ms = t0.elapsed().as_millis() as u64;

    let t1 = Instant::now();
    Command::new("avifdec").arg(&enc_out).arg(&dec_out).output()?;
    let decode_ms = t1.elapsed().as_millis() as u64;

    let size_bytes = std::fs::metadata(&enc_out)?.len();
    let (w, h, ref_lms) = load_lms(input)?;
    let (_, _, dec_lms) = load_lms(&dec_out)?;
    let d = mean_d_oklab(w, h, &ref_lms, &dec_lms);

    Ok(CodecResult { codec: "avif", size_bytes, encode_ms, decode_ms, d_oklab: d })
}

/// Run JXL encode/decode and measure.
pub fn bench_jxl(input: &Path, quality: f32) -> Result<CodecResult> {
    let enc_out = std::env::temp_dir().join("cifv2_cmp.jxl");
    let dec_out = std::env::temp_dir().join("cifv2_cmp_jxl.png");

    let t0 = Instant::now();
    Command::new("cjxl")
        .arg(input).arg(&enc_out)
        .args(["--quality", &quality.to_string()])
        .output()?;
    let encode_ms = t0.elapsed().as_millis() as u64;

    let t1 = Instant::now();
    Command::new("djxl").arg(&enc_out).arg(&dec_out).output()?;
    let decode_ms = t1.elapsed().as_millis() as u64;

    let size_bytes = std::fs::metadata(&enc_out)?.len();
    let (w, h, ref_lms) = load_lms(input)?;
    let (_, _, dec_lms) = load_lms(&dec_out)?;
    let d = mean_d_oklab(w, h, &ref_lms, &dec_lms);

    Ok(CodecResult { codec: "jxl", size_bytes, encode_ms, decode_ms, d_oklab: d })
}

/// Run WebP encode/decode and measure.
pub fn bench_webp(input: &Path, quality: u32) -> Result<CodecResult> {
    let enc_out = std::env::temp_dir().join("cifv2_cmp.webp");
    let dec_out = std::env::temp_dir().join("cifv2_cmp_webp.png");

    let t0 = Instant::now();
    Command::new("cwebp")
        .args(["-q", &quality.to_string()])
        .arg(input).args(["-o", enc_out.to_str().unwrap()])
        .output()?;
    let encode_ms = t0.elapsed().as_millis() as u64;

    let t1 = Instant::now();
    Command::new("dwebp")
        .arg(&enc_out).args(["-o", dec_out.to_str().unwrap()])
        .output()?;
    let decode_ms = t1.elapsed().as_millis() as u64;

    let size_bytes = std::fs::metadata(&enc_out)?.len();
    let (w, h, ref_lms) = load_lms(input)?;
    let (_, _, dec_lms) = load_lms(&dec_out)?;
    let d = mean_d_oklab(w, h, &ref_lms, &dec_lms);

    Ok(CodecResult { codec: "webp", size_bytes, encode_ms, decode_ms, d_oklab: d })
}
