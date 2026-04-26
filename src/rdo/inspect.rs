//! CIF-RDO artifact inspector.
//!
//! Decodes each tile, recomputes D_oklab against the source image,
//! and reports per-tile diagnostics.

use anyhow::Result;
use std::path::Path;
use std::fs;
use image::GenericImageView;

use crate::rdo::types::{LmsTile, TILE_SIZE, FIX_SCALE};
use crate::rdo::objective::d_oklab_fixed;
use crate::rdo::tree_bin::{read_tree, encoder_name};
use crate::rdo::encoders::constant::ConstantLms;
use crate::rdo::encoders::affine::AffineLms;
use crate::rdo::encoders::quadratic::QuadraticLms;
use crate::rdo::encoders::wavelet::WaveletTile;
use crate::rdo::encoders::dct::DctTile;
use crate::rdo::encoders::edge::EdgeTile;
use crate::rdo::encoders::siren::MicroSirenTile;
use crate::rdo::types::RegionEncoder;

fn get_encoder(name: &str) -> Box<dyn RegionEncoder> {
    match name {
        "constant_lms"    => Box::new(ConstantLms),
        "affine_lms"      => Box::new(AffineLms),
        "quadratic_lms"   => Box::new(QuadraticLms),
        "wavelet_tile"    => Box::new(WaveletTile::new(1)),
        "dct_tile"        => Box::new(DctTile::new(FIX_SCALE)),
        "edge_tile"       => Box::new(EdgeTile),
        "micro_siren_tile"=> Box::new(MicroSirenTile),
        other => panic!("unknown encoder: {other}"),
    }
}

fn srgb_to_linear(v: u8) -> f32 { (v as f32 / 255.0).powf(2.2) }

pub fn rdo_inspect(artifact: &Path, source: Option<&Path>) -> Result<()> {
    let tree_bytes       = fs::read(artifact.join("regions/tree.bin"))?;
    let payload_compressed = fs::read(artifact.join("regions/payloads.bin.zst"))?;
    let payload_bytes    = zstd::decode_all(payload_compressed.as_slice())?;

    let (hdr, tile_records) = read_tree(&tree_bytes)?;
    let tw     = hdr.tile_size as usize;
    let grid_w = hdr.grid_w as usize;
    let grid_h = hdr.grid_h as usize;
    let vis_w  = hdr.visible_w as usize;
    let vis_h  = hdr.visible_h as usize;

    // Load source image for reference distortion if provided
    let ref_lms: Option<Vec<[i64;3]>> = if let Some(src_path) = source {
        let img = image::open(src_path)?.to_rgba8();
        let (iw, ih) = img.dimensions();
        let mut buf = Vec::with_capacity((iw*ih) as usize);
        for p in img.pixels() {
            let r = srgb_to_linear(p[0]);
            let g = srgb_to_linear(p[1]);
            let b = srgb_to_linear(p[2]);
            let l = 0.31399*r + 0.63951*g + 0.04649*b;
            let m = 0.15537*r + 0.75789*g + 0.08670*b;
            let s = 0.01775*r + 0.10944*g + 0.87257*b;
            buf.push([
                (l * FIX_SCALE as f32).round() as i64,
                (m * FIX_SCALE as f32).round() as i64,
                (s * FIX_SCALE as f32).round() as i64,
            ]);
        }
        Some(buf)
    } else { None };

    // Header
    println!("artifact: {}", artifact.display());
    println!("grid: {}x{} tiles, visible: {}x{}, tile_size: {}",
        grid_w, grid_h, vis_w, vis_h, tw);
    println!("quality_lambda: {}", hdr.quality_lambda);
    println!();
    println!("{:>4}  {:<18} {:>5}  {:>10}  {:>10}  bounds",
        "id", "encoder", "bytes", "D_oklab", "J_fixed");
    println!("{}", "-".repeat(72));

    let mut payload_offset = 0usize;
    let mut total_d = 0i64;
    let mut worst_d = 0i64;
    let mut worst_id = 0usize;

    for (idx, rec) in tile_records.iter().enumerate() {
        let gx = idx % grid_w;
        let gy = idx / grid_w;
        let enc_name = encoder_name(rec.encoder_id);
        let len = rec.payload_len as usize;

        // Decode tile
        let encoder = get_encoder(enc_name);
        let encoded = crate::rdo::types::EncodedRegion {
            encoder_name: encoder.name(),
            payload: payload_bytes[payload_offset..payload_offset+len].to_vec(),
            rate_bits: (len*8) as u64,
            decode_cost: 0, encode_cost: 0, memory_cost: 0, replay_cost: 0,
        };
        let mut decoded = LmsTile::new(gx, gy);
        encoder.decode(&encoded, &mut decoded);

        // Compute D_oklab against source if available
        let d = if let Some(ref lms) = ref_lms {
            let pad_w = grid_w * tw;
            let mut ref_tile = LmsTile::new(gx, gy);
            for ty in 0..tw {
                for tx in 0..tw {
                    let px = (gx*tw+tx).min(vis_w-1);
                    let py = (gy*tw+ty).min(vis_h-1);
                    let p = lms[py*vis_w+px];  // use vis_w not pad_w
                    ref_tile.set(tx, ty, 0, p[0]);
                    ref_tile.set(tx, ty, 1, p[1]);
                    ref_tile.set(tx, ty, 2, p[2]);
                }
            }
            d_oklab_fixed(&ref_tile, &decoded)
        } else { 0 };

        let j = (len as i64 * 8).saturating_mul(FIX_SCALE)
            .saturating_add(hdr.quality_lambda as i64 * d);

        let d_float = d as f64 / FIX_SCALE as f64;
        let x0 = gx*tw; let y0 = gy*tw;
        let x1 = (x0+tw).min(vis_w);
        let y1 = (y0+tw).min(vis_h);

        println!("{:>4}  {:<18} {:>5}  {:>10.6}  {:>10}  [{},{},{},{}]",
            idx, enc_name, len, d_float, j, x0, y0, x1, y1);

        total_d += d;
        payload_offset += len;

        if d > worst_d { worst_d = d; worst_id = idx; }
    }

    let n = tile_records.len();
    println!("{}", "-".repeat(72));
    println!("mean D_oklab: {:.6}", total_d as f64 / n as f64 / FIX_SCALE as f64);
    println!("worst tile:   id={worst_id}  D_oklab={:.6}",
        worst_d as f64 / FIX_SCALE as f64);

    Ok(())
}
