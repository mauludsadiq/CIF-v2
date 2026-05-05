//! CIF-RDO v0 benchmark command.
//!
//! Encodes all images in a directory and reports per-image encoder distribution,
//! artifact size, and encode time.

use anyhow::Result;
use std::path::Path;
use std::fs;
use std::time::Instant;

pub fn rdo_bench(corpus: &Path, quality: f64, compare: bool) -> Result<()> {
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
    println!("{:<22} {:>5} {:>6} {:>6} {:>5} {:>5} {:>5} {:>5} {:>5} {:>7} {:>6}",
        "image", "tiles", "const", "affine", "quad", "wave", "edge", "siren", "dct", "size_kb", "ms");
    println!("{}", "-".repeat(80));

    let mut totals = [0usize; 7];
    let mut total_size = 0u64;
    let mut total_ms = 0u64;
    let mut total_tiles = 0usize;

    for entry in &entries {
        let input = entry.path();
        let name = input.file_stem().unwrap().to_string_lossy().to_string();
        let out = std::env::temp_dir().join(format!("cifv2_bench_{name}.cifrdo"));
        // Remove stale artifact before re-encoding
        let _ = std::fs::remove_dir_all(&out);

        let t0 = Instant::now();
        std::env::set_var("CIFV2_QUIET", "1");
        crate::rdo::encode::rdo_encode(&input, &out, 32, quality)?;
        std::env::remove_var("CIFV2_QUIET");
        let ms = t0.elapsed().as_millis() as u64;

        let tree_bytes = fs::read(out.join("regions/tree.bin"))?;
        let (_hdr, tile_records) = crate::rdo::tree_bin::read_tree(&tree_bytes)?;
        let n = tile_records.len();

        let mut enc = [0usize; 7];
        for r in &tile_records {
            let i = r.encoder_id as usize;
            if i < enc.len() {
                enc[i] += 1;
            }
        }

        // Artifact size
        let size = fs::read(out.join("regions/payloads.bin.zst"))?.len() as u64
            + fs::read(out.join("regions/tree.bin"))?.len() as u64
            + fs::read(out.join("manifest.json"))?.len() as u64
            + fs::read(out.join("receipt.json"))?.len() as u64;
        let size_kb = (size + 512) / 1024;

        println!("{:<22} {:>5} {:>6} {:>6} {:>5} {:>5} {:>5} {:>5} {:>5} {:>7} {:>6}",
            &name[..name.len().min(22)], n,
            enc[0], enc[1], enc[2], enc[3], enc[4], enc[5], enc[6],
            size_kb, ms);

        for i in 0..7 { totals[i] += enc[i]; }
        total_size += size;
        total_ms += ms;
        total_tiles += n;
    }

    println!("{}", "-".repeat(80));
    println!("{:<22} {:>5} {:>6} {:>6} {:>5} {:>5} {:>5} {:>5} {:>5} {:>7} {:>6}",
        "TOTAL", total_tiles,
        totals[0], totals[1], totals[2], totals[3], totals[4], totals[5], totals[6],
        (total_size + 512) / 1024, total_ms);

    if compare {
        println!();
        println!("External codec comparison (quality: avif=60 jxl=60 webp=60):");
        println!("{:<22} {:<8} {:>8} {:>10} {:>10} {:>10}",
            "image", "codec", "size_kb", "encode_ms", "decode_ms", "D_oklab");
        println!("{}", "-".repeat(72));

        for entry in &entries {
            let input = entry.path();
            let name = input.file_stem().unwrap().to_string_lossy().to_string();
            let short = &name[..name.len().min(22)];

            // CIF-RDO result — render and measure
            let cifrdo_out = std::env::temp_dir().join(format!("cifv2_bench_{name}.cifrdo"));
            let render_out = std::env::temp_dir().join(format!("cifv2_bench_{name}_render.png"));
            std::env::set_var("CIFV2_QUIET", "1");
            crate::rdo::encode::rdo_encode(&input, &cifrdo_out, 32, quality)?;
            std::env::remove_var("CIFV2_QUIET");
            std::env::set_var("CIFV2_QUIET", "1");
            crate::rdo::render::rdo_render(&cifrdo_out, &render_out, 256, 256)?;
            std::env::remove_var("CIFV2_QUIET");

            let (w, h, ref_lms) = crate::rdo::compare::load_lms(&input)?;
            let (_, _, rdo_lms) = crate::rdo::compare::load_lms(&render_out)?;
            let rdo_d = crate::rdo::compare::mean_d_oklab(w, h, &ref_lms, &rdo_lms);
            let rdo_size = {
                let t = &cifrdo_out;
                fs::read(t.join("regions/payloads.bin.zst"))?.len() as u64
                    + fs::read(t.join("regions/tree.bin"))?.len() as u64
                    + fs::read(t.join("manifest.json"))?.len() as u64
            };
            println!("{:<22} {:<8} {:>8} {:>10} {:>10} {:>10.6}",
                short, "cif-rdo", (rdo_size+512)/1024, "-", "-", rdo_d);

            // External codecs
            for result in [
                crate::rdo::compare::bench_avif(&input, 60),
                crate::rdo::compare::bench_jxl(&input, 60.0),
                crate::rdo::compare::bench_webp(&input, 60),
            ] {
                match result {
                    Ok(r) => println!("{:<22} {:<8} {:>8} {:>10} {:>10} {:>10.6}",
                        "", r.codec, (r.size_bytes+512)/1024,
                        r.encode_ms, r.decode_ms, r.d_oklab),
                    Err(e) => println!("  error: {e}"),
                }
            }
            println!();
        }
    }

    Ok(())
}
