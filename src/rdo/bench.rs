//! CIF-RDO v0 benchmark command.
//!
//! Encodes all images in a directory and reports per-image encoder distribution,
//! artifact size, and encode time.

use anyhow::Result;
use std::path::Path;
use std::fs;
use std::time::Instant;
use serde_json::Value;

pub fn rdo_bench(corpus: &Path, quality: f64) -> Result<()> {
    let mut entries: Vec<_> = fs::read_dir(corpus)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let p = e.path();
            matches!(p.extension().and_then(|s| s.to_str()), Some("png"|"jpg"|"jpeg"))
        })
        .collect();
    entries.sort_by_key(|e| e.path());

    if entries.is_empty() {
        println!("No images found in {}", corpus.display());
        return Ok(());
    }

    // Header
    println!("{:<22} {:>5} {:>6} {:>6} {:>5} {:>5} {:>5} {:>5} {:>7} {:>6}",
        "image", "tiles", "const", "affine", "quad", "wave", "edge", "siren", "size_kb", "ms");
    println!("{}", "-".repeat(80));

    let mut totals = [0usize; 6];
    let mut total_size = 0u64;
    let mut total_ms = 0u64;
    let mut total_tiles = 0usize;

    for entry in &entries {
        let input = entry.path();
        let name = input.file_stem().unwrap().to_string_lossy().to_string();
        let out = std::env::temp_dir().join(format!("cifv2_bench_{name}.cifrdo"));

        let t0 = Instant::now();
        std::env::set_var("CIFV2_QUIET", "1");
        crate::rdo::encode::rdo_encode(&input, &out, 32, quality)?;
        std::env::remove_var("CIFV2_QUIET");
        let ms = t0.elapsed().as_millis() as u64;

        let tree: Value = serde_json::from_slice(&fs::read(out.join("regions/tree.json"))?)?;
        let regions = tree["regions"].as_array().unwrap();
        let n = regions.len();

        let mut enc = [0usize; 6];
        let names = ["constant_lms","affine_lms","quadratic_lms","wavelet_tile","edge_tile","micro_siren_tile"];
        for r in regions {
            let e = r["encoder"].as_str().unwrap_or("");
            if let Some(i) = names.iter().position(|&n| n == e) {
                enc[i] += 1;
            }
        }

        // Artifact size
        let size = fs::read(out.join("regions/payloads.bin"))?.len() as u64
            + fs::read(out.join("regions/tree.json"))?.len() as u64
            + fs::read(out.join("manifest.json"))?.len() as u64
            + fs::read(out.join("receipt.json"))?.len() as u64;
        let size_kb = (size + 512) / 1024;

        println!("{:<22} {:>5} {:>6} {:>6} {:>5} {:>5} {:>5} {:>5} {:>7} {:>6}",
            &name[..name.len().min(22)], n,
            enc[0], enc[1], enc[2], enc[3], enc[4], enc[5],
            size_kb, ms);

        for i in 0..6 { totals[i] += enc[i]; }
        total_size += size;
        total_ms += ms;
        total_tiles += n;
    }

    println!("{}", "-".repeat(80));
    println!("{:<22} {:>5} {:>6} {:>6} {:>5} {:>5} {:>5} {:>5} {:>7} {:>6}",
        "TOTAL", total_tiles,
        totals[0], totals[1], totals[2], totals[3], totals[4], totals[5],
        (total_size + 512) / 1024, total_ms);

    Ok(())
}
