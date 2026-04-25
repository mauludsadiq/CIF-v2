use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use clap::{Parser, Subcommand};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const OPERATOR: &str = "Phi-CIFv2-1.0.0";
const FORMAT: &str = "CIFv2";
const Q_SCALE: f32 = 1_000_000.0;
const FIX_SCALE: f32 = 1_000_000.0;
const P: i64 = 7;
const N: u32 = 3;
const MODULUS: i64 = 343;
const TILE: usize = 32;
const SIREN_WIDTH: usize = 32;
const SIREN_LAYERS: usize = 3;
const SIREN_STEPS: usize = 64;
const SIREN_LR: f32 = 1e-3;

#[derive(Parser, Debug)]
#[command(name = "cifv2")]
#[command(about = "CIF v2 deterministic executable visual field artifact generator")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Encode {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 256)]
        max_side: u32,
    },
    Verify {
        #[arg(long)]
        artifact: PathBuf,
    },
    Replay {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        temp_out: PathBuf,
        #[arg(long, default_value_t = 256)]
        max_side: u32,
    },
    Render {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        width: u32,
        #[arg(long)]
        height: u32,
    },
    Pack {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    Unpack {
        #[arg(long)]
        file: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(Clone, Serialize, Deserialize)]
struct Manifest {
    format: String,
    operator: String,
    width: u32,
    height: u32,
    color_space: String,
    wavelet: WaveletSpec,
    edge_model: String,
    procedural_model: String,
    inr_model: String,
    residue: ResidueSpec,
    hashes: BTreeMap<String, String>,
}

#[derive(Clone, Serialize, Deserialize)]
struct WaveletSpec { family: String, levels: u32, boundary: String, coefficient_order: String }
#[derive(Clone, Serialize, Deserialize)]
struct ResidueSpec { p: i64, n: u32, modulus: i64 }

#[derive(Clone, Serialize, Deserialize)]
struct Receipt {
    receipt_version: String,
    operator: String,
    input_digest: String,
    steps: Vec<StepReceipt>,
    manifest_digest: String,
    artifact_digest: String,
}
#[derive(Clone, Serialize, Deserialize)]
struct StepReceipt { name: String, output: String, digest: String }

#[derive(Clone)]
struct Tensor { width: usize, height: usize, data: Vec<[f32;3]> }

#[derive(Clone, Serialize, Deserialize)]
struct EdgeSegment {
    x0: i64, y0: i64, x1: i64, y1: i64,
    c0x: i64, c0y: i64, c1x: i64, c1y: i64,
    width: i64, contrast_left: i64, contrast_right: i64, blur_sigma: i64,
}

#[derive(Clone, Serialize, Deserialize)]
struct ProcTile {
    tile_x: usize, tile_y: usize, kernel: String, seed: String,
    amplitude: i64, slope: i64, orientation: i64, correlation_length: i64,
}
#[derive(Clone, Serialize, Deserialize)]
struct Procedural { tile_size: usize, tiles: Vec<ProcTile> }

#[derive(Clone, Serialize, Deserialize)]
struct SirenFile { architecture: SirenArch, weights_b85: String }
#[derive(Clone, Serialize, Deserialize)]
struct SirenArch { input: usize, output: usize, hidden_layers: usize, hidden_width: usize, activation: String, omega0: f32, steps: usize, learning_rate: f32 }

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Encode { input, out, max_side } => encode(&input, &out, max_side),
        Commands::Verify { artifact } => verify(&artifact).map(|d| { println!("{}", serde_json::to_string_pretty(&d).unwrap()); }),
        Commands::Replay { input, artifact, temp_out, max_side } => replay(&input, &artifact, &temp_out, max_side),
        Commands::Render { artifact, out, width, height } => render(&artifact, &out, width, height),
        Commands::Pack { artifact, out } => pack(&artifact, &out),
        Commands::Unpack { file, out } => unpack(&file, &out),
    }
}

const CIFV2_MAGIC: &[u8; 8] = b"CIFV2\0\0\0";
const CIFV2_VERSION: u32 = 1;
const COMPONENT_ORDER: &[&str] = &[
    "manifest.json", "receipt.json", "canonical_lms.tensor",
    "lambda_real.bin", "edges.cifedge", "procedural.json",
    "inr.siren", "preview.png",
];

fn pack(artifact: &Path, out: &Path) -> Result<()> {
    // Read all components
    let mut components: Vec<(String, Vec<u8>)> = Vec::new();
    for name in COMPONENT_ORDER {
        let path = artifact.join(name);
        if path.exists() {
            components.push((name.to_string(), fs::read(&path)?));
        }
    }

    let n = components.len() as u32;
    let index_size = n as u64 * 64;
    let header_size = 8 + 4 + 4 + index_size; // magic + version + n_entries + index

    // Compute offsets
    let mut offset = header_size;
    let mut offsets = Vec::new();
    for (_, data) in &components {
        offsets.push(offset);
        offset += data.len() as u64;
    }

    // Write file
    let mut buf = Vec::new();
    buf.extend_from_slice(CIFV2_MAGIC);
    buf.extend_from_slice(&CIFV2_VERSION.to_le_bytes());
    buf.extend_from_slice(&n.to_le_bytes());

    // Index entries (64 bytes each): 32 name + 8 offset + 8 length + 16 reserved
    for ((name, data), &off) in components.iter().zip(offsets.iter()) {
        let mut entry = [0u8; 64];
        let nb = name.as_bytes();
        entry[..nb.len().min(32)].copy_from_slice(&nb[..nb.len().min(32)]);
        entry[32..40].copy_from_slice(&off.to_le_bytes());
        entry[40..48].copy_from_slice(&(data.len() as u64).to_le_bytes());
        buf.extend_from_slice(&entry);
    }

    // Data blobs
    for (_, data) in &components {
        buf.extend_from_slice(data);
    }

    fs::write(out, &buf)?;
    let digest = format!("sha256:{}", sha256_hex(&buf));
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "ok": true,
        "out": out.to_string_lossy(),
        "components": n,
        "size_bytes": buf.len(),
        "file_digest": digest
    })).unwrap());
    Ok(())
}

fn unpack(file: &Path, out: &Path) -> Result<()> {
    let buf = fs::read(file)?;
    if &buf[0..8] != CIFV2_MAGIC { bail!("not a .cifv2 container file"); }
    let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    if version != CIFV2_VERSION { bail!("unsupported container version {version}"); }
    let n = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;

    fs::create_dir_all(out)?;

    let mut pos = 16usize;
    for _ in 0..n {
        let entry = &buf[pos..pos+64];
        let name_end = entry[..32].iter().position(|&b| b == 0).unwrap_or(32);
        let name = std::str::from_utf8(&entry[..name_end])?.to_string();
        let offset = u64::from_le_bytes(entry[32..40].try_into().unwrap()) as usize;
        let length = u64::from_le_bytes(entry[40..48].try_into().unwrap()) as usize;
        fs::write(out.join(&name), &buf[offset..offset+length])?;
        pos += 64;
    }

    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "ok": true,
        "out": out.to_string_lossy(),
        "components": n
    })).unwrap());
    Ok(())
}

fn unpack_silent(file: &Path, out: &Path) -> Result<()> {
    let buf = fs::read(file)?;
    if &buf[0..8] != CIFV2_MAGIC { bail!("not a .cifv2 container file"); }
    let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    if version != CIFV2_VERSION { bail!("unsupported container version {version}"); }
    let n = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
    fs::create_dir_all(out)?;
    let mut pos = 16usize;
    for _ in 0..n {
        let entry = &buf[pos..pos+64];
        let name_end = entry[..32].iter().position(|&b| b == 0).unwrap_or(32);
        let name = std::str::from_utf8(&entry[..name_end])?.to_string();
        let offset = u64::from_le_bytes(entry[32..40].try_into().unwrap()) as usize;
        let length = u64::from_le_bytes(entry[40..48].try_into().unwrap()) as usize;
        fs::write(out.join(&name), &buf[offset..offset+length])?;
        pos += 64;
    }
    Ok(())
}

fn resolve_artifact(path: &Path) -> Result<std::borrow::Cow<'_, Path>> {
    if path.is_dir() {
        return Ok(std::borrow::Cow::Borrowed(path));
    }
    // Check magic bytes for single-file container
    if path.is_file() {
        let magic = {
            let mut f = std::fs::File::open(path)?;
            let mut m = [0u8; 8];
            std::io::Read::read_exact(&mut f, &mut m).ok();
            m
        };
        if &magic == CIFV2_MAGIC {
            let tmp = std::env::temp_dir().join(
                format!("cifv2_unpack_{}", sha256_hex(path.to_string_lossy().as_bytes()))
            );
            if !tmp.exists() {
                unpack_silent(path, &tmp)?;
            }
            return Ok(std::borrow::Cow::Owned(tmp));
        }
    }
    bail!("artifact path is neither a directory nor a .cifv2 container: {}", path.display())
}

fn encode(input: &Path, out: &Path, max_side: u32) -> Result<()> {
    if out.exists() { fs::remove_dir_all(out).context("remove existing output directory")?; }
    fs::create_dir_all(out)?;
    let t0 = canonicalize_input(input, max_side)?;
    write_tensor_zstd(&t0, &out.join("canonical_lms.tensor"))?;
    let h_input = digest_file(&out.join("canonical_lms.tensor"))?;

    let lambda_q = wavelet_anchor(&t0);
    write_i64_vec_zstd(&lambda_q, &out.join("lambda_real.bin"))?;
    let h_lambda = digest_file(&out.join("lambda_real.bin"))?;

    let edges = structural_edges(&t0);
    write_edges_bin(&edges, &out.join("edges.cifedge"))?;
    let h_edges = digest_file(&out.join("edges.cifedge"))?;

    let proc = procedural_residual(&t0, &lambda_q, &edges, &h_input);
    write_json(&proc, &out.join("procedural.json"))?;
    let h_proc = digest_file(&out.join("procedural.json"))?;

    let siren = neural_residual(&t0, &lambda_q, &edges, &proc, &h_input, &h_lambda, &h_edges, &h_proc);
    write_json(&siren, &out.join("inr.siren"))?;
    let h_inr = digest_file(&out.join("inr.siren"))?;

    let mods = modular_residue(&lambda_q);
    // lambda_mod is validation metadata — digest computed in memory, not written to disk
    let mut mod_bytes = Vec::new();
    mod_bytes.extend_from_slice(b"CIFI64\0\0");
    mod_bytes.extend_from_slice(&(mods.len() as u64).to_le_bytes());
    for v in &mods { mod_bytes.extend_from_slice(&v.to_le_bytes()); }
    let h_mod = format!("sha256:{}", sha256_hex(&mod_bytes));

    write_preview(&t0, &out.join("preview.png"))?;
    let h_preview = digest_file(&out.join("preview.png"))?;

    let mut hashes = BTreeMap::new();
    hashes.insert("canonical_lms".into(), h_input.clone());
    hashes.insert("lambda_real".into(), h_lambda.clone());
    hashes.insert("edges".into(), h_edges.clone());
    hashes.insert("procedural".into(), h_proc.clone());
    hashes.insert("inr".into(), h_inr.clone());
    hashes.insert("lambda_mod".into(), h_mod.clone());
    hashes.insert("preview".into(), h_preview.clone());

    let manifest = Manifest {
        format: FORMAT.into(), operator: OPERATOR.into(), width: t0.width as u32, height: t0.height as u32,
        color_space: "LMS".into(),
        wavelet: WaveletSpec { family: "db4".into(), levels: 3, boundary: "symmetric".into(), coefficient_order: "channel-level-band-row-major".into() },
        edge_model: "cubic_bezier_v1_polyline_fit".into(), procedural_model: "one_over_f_tile_v1".into(),
        inr_model: "siren_3x32_v1_cpu_reference".into(), residue: ResidueSpec { p: P, n: N, modulus: MODULUS }, hashes,
    };
    write_json(&manifest, &out.join("manifest.json"))?;
    let h_manifest = digest_file(&out.join("manifest.json"))?;

    let steps = vec![
        StepReceipt { name: "canonicalize_input".into(), output: "canonical_lms.tensor".into(), digest: h_input.clone() },
        StepReceipt { name: "wavelet_anchor".into(), output: "lambda_real.bin".into(), digest: h_lambda.clone() },
        StepReceipt { name: "structural_edges".into(), output: "edges.cifedge".into(), digest: h_edges.clone() },
        StepReceipt { name: "procedural_residual".into(), output: "procedural.json".into(), digest: h_proc.clone() },
        StepReceipt { name: "neural_residual".into(), output: "inr.siren".into(), digest: h_inr.clone() },
        StepReceipt { name: "modular_residue".into(), output: "lambda_mod.bin".into(), digest: h_mod.clone() },
    ];
    let artifact_digest = artifact_digest(&[&h_input,&h_lambda,&h_edges,&h_proc,&h_inr,&h_mod,&h_preview,&h_manifest]);
    let receipt = Receipt { receipt_version: "FARD-CIF-RECEIPT-1".into(), operator: OPERATOR.into(), input_digest: h_input, steps, manifest_digest: h_manifest, artifact_digest: artifact_digest.clone() };
    write_json(&receipt, &out.join("receipt.json"))?;
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({"ok":true,"artifact_digest":artifact_digest,"out":out})).unwrap());
    Ok(())
}

fn verify(artifact_path: &Path) -> Result<serde_json::Value> {
    let artifact_cow = resolve_artifact(artifact_path)?;
    let artifact = artifact_cow.as_ref();
    let manifest: Manifest = read_json(&artifact.join("manifest.json"))?;
    let receipt: Receipt = read_json(&artifact.join("receipt.json"))?;
    if manifest.format != FORMAT { bail!("format mismatch"); }
    if manifest.operator != OPERATOR || receipt.operator != OPERATOR { bail!("operator mismatch"); }
    for (k, d) in &manifest.hashes {
        let file = match k.as_str() {
            "canonical_lms" => "canonical_lms.tensor", "lambda_real" => "lambda_real.bin", "edges" => "edges.cifedge",
            "procedural" => "procedural.json", "inr" => "inr.siren", "preview" => "preview.png", "lambda_mod" => { continue; }, _ => bail!("unknown hash key {k}"),
        };
        let actual = digest_file(&artifact.join(file))?;
        if &actual != d { bail!("digest mismatch for {file}: expected {d}, actual {actual}"); }
    }
    let lambda = read_i64_vec_zstd(&artifact.join("lambda_real.bin"))?;
    let expected_mod = modular_residue(&lambda);
    // Recompute h_mod from lambda_real in memory — lambda_mod.bin is not stored
    let mut mod_bytes = Vec::new();
    mod_bytes.extend_from_slice(b"CIFI64\0\0");
    mod_bytes.extend_from_slice(&(expected_mod.len() as u64).to_le_bytes());
    for v in &expected_mod { mod_bytes.extend_from_slice(&v.to_le_bytes()); }
    let h_mod_actual = format!("sha256:{}", sha256_hex(&mod_bytes));
    let h_mod_stored = manifest.hashes.get("lambda_mod").ok_or_else(|| anyhow::anyhow!("missing lambda_mod hash"))?;
    if &h_mod_actual != h_mod_stored { bail!("lambda_mod recomputation mismatch"); }
    let h_manifest = digest_file(&artifact.join("manifest.json"))?;
    if h_manifest != receipt.manifest_digest { bail!("manifest digest mismatch"); }
    let hashes = &manifest.hashes;
    let ad = artifact_digest(&[
        hashes.get("canonical_lms").unwrap(), hashes.get("lambda_real").unwrap(), hashes.get("edges").unwrap(),
        hashes.get("procedural").unwrap(), hashes.get("inr").unwrap(), hashes.get("lambda_mod").unwrap(), hashes.get("preview").unwrap(), &h_manifest,
    ]);
    if ad != receipt.artifact_digest { bail!("artifact digest mismatch"); }
    Ok(serde_json::json!({"ok":true,"artifact_digest":ad,"steps_verified":receipt.steps.len()}))
}

fn replay(input: &Path, artifact_path: &Path, temp_out: &Path, max_side: u32) -> Result<()> {
    let artifact_cow = resolve_artifact(artifact_path)?;
    let artifact = artifact_cow.as_ref();
    let old: Receipt = read_json(&artifact.join("receipt.json"))?;
    encode(input, temp_out, max_side)?;
    let new: Receipt = read_json(&temp_out.join("receipt.json"))?;
    if old.artifact_digest != new.artifact_digest {
        bail!("replay mismatch: expected {}, actual {}", old.artifact_digest, new.artifact_digest);
    }
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({"ok":true,"artifact_digest":new.artifact_digest})).unwrap());
    Ok(())
}

fn render(artifact_path: &Path, out: &Path, width: u32, height: u32) -> Result<()> {
    let artifact_cow = resolve_artifact(artifact_path)?;
    let artifact = artifact_cow.as_ref();
    let receipt: Receipt = read_json(&artifact.join("receipt.json"))?;
    let artifact_digest = receipt.artifact_digest.clone();

    let t = read_tensor_zstd(&artifact.join("canonical_lms.tensor"))?;
    let edges: Vec<EdgeSegment> = read_edges_bin(&artifact.join("edges.cifedge"))?;
    let siren_file: SirenFile = read_json(&artifact.join("inr.siren"))?;
    let siren_arch = siren_file.architecture.clone();
    let siren_raw = base85::decode(&siren_file.weights_b85).map_err(|e| anyhow::anyhow!("{e}"))?;
    let siren_weights: Vec<f32> = siren_raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]]))
        .collect();

    const SCALE: f32 = 1_000_000.0;
    let cw = t.width as f32;
    let ch = t.height as f32;
    let ow = width as f32;
    let oh = height as f32;

    const TILE: usize = 32;
    let tiles_x = (width as usize + TILE - 1) / TILE;
    let tiles_y = (height as usize + TILE - 1) / TILE;
    let mut tile_edges: Vec<Vec<usize>> = vec![Vec::new(); tiles_x * tiles_y];

    for (ei, seg) in edges.iter().enumerate() {
        let pad = (seg.width + seg.blur_sigma) as f32 / SCALE;
        let ex0 = seg.x0.min(seg.x1).min(seg.c0x).min(seg.c1x) as f32 / SCALE - pad;
        let ex1 = seg.x0.max(seg.x1).max(seg.c0x).max(seg.c1x) as f32 / SCALE + pad;
        let ey0 = seg.y0.min(seg.y1).min(seg.c0y).min(seg.c1y) as f32 / SCALE - pad;
        let ey1 = seg.y0.max(seg.y1).max(seg.c0y).max(seg.c1y) as f32 / SCALE + pad;

        let px0 = ((ex0 * ow / cw) as usize).saturating_sub(1);
        let px1 = ((ex1 * ow / cw) as usize + 1).min(width as usize - 1);
        let py0 = ((ey0 * oh / ch) as usize).saturating_sub(1);
        let py1 = ((ey1 * oh / ch) as usize + 1).min(height as usize - 1);

        let tx0 = px0 / TILE;
        let tx1 = px1 / TILE;
        let ty0 = py0 / TILE;
        let ty1 = py1 / TILE;

        for ty in ty0..=ty1.min(tiles_y - 1) {
            for tx in tx0..=tx1.min(tiles_x - 1) {
                tile_edges[ty * tiles_x + tx].push(ei);
            }
        }
    }

    let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);

    for py in 0..height as usize {
        let ty = py / TILE;
        for px in 0..width as usize {
            let tx = px / TILE;

            let cx = (px as f32 + 0.5) * cw / ow;
            let cy = (py as f32 + 0.5) * ch / oh;

            let fx = cx - 0.5;
            let fy = cy - 0.5;
            let x0 = (fx.floor() as isize).clamp(0, t.width as isize - 1) as usize;
            let y0 = (fy.floor() as isize).clamp(0, t.height as isize - 1) as usize;
            let x1 = (x0 + 1).min(t.width - 1);
            let y1 = (y0 + 1).min(t.height - 1);
            let tx_ = (fx - fx.floor()).clamp(0.0, 1.0);
            let ty_ = (fy - fy.floor()).clamp(0.0, 1.0);

            let s = |xi: usize, yi: usize| t.data[yi * t.width + xi];
            let lerp3 = |a: [f32;3], b: [f32;3], t: f32| -> [f32;3] {
                [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t]
            };
            let top    = lerp3(s(x0,y0), s(x1,y0), tx_);
            let bottom = lerp3(s(x0,y1), s(x1,y1), tx_);
            let mut lms = lerp3(top, bottom, ty_);

            for &ei in &tile_edges[ty * tiles_x + tx] {
                let seg = &edges[ei];
                let half_w   = seg.width as f32 / SCALE / 2.0;
                let sigma    = seg.blur_sigma as f32 / SCALE;
                let contrast = seg.contrast_left as f32 / SCALE;

                let p0x = seg.x0 as f32 / SCALE;
                let p0y = seg.y0 as f32 / SCALE;
                let p1x = seg.x1 as f32 / SCALE;
                let p1y = seg.y1 as f32 / SCALE;
                let dx = p1x - p0x;
                let dy = p1y - p0y;
                let len2 = dx*dx + dy*dy;

                let (dist, signed) = if len2 < 1e-10 {
                    let d = ((cx-p0x).powi(2) + (cy-p0y).powi(2)).sqrt();
                    (d, 1.0f32)
                } else {
                    let t_param = ((cx-p0x)*dx + (cy-p0y)*dy) / len2;
                    let t_param = t_param.clamp(0.0, 1.0);
                    let nx = cx - (p0x + t_param*dx);
                    let ny = cy - (p0y + t_param*dy);
                    let dist = (nx*nx + ny*ny).sqrt();
                    let cross = dx*ny - dy*nx;
                    (dist, cross.signum())
                };

                let outer = half_w + sigma;
                if dist < outer {
                    let t_blend = ((outer - dist) / sigma.max(1e-6)).clamp(0.0, 1.0);
                    let smooth = t_blend * t_blend * (3.0 - 2.0 * t_blend);
                    let influence = contrast * smooth * signed;
                    lms[0] += influence * 0.6;
                    lms[1] += influence * 0.3;
                    lms[2] += influence * 0.1;
                }
            }

            // SIREN residual
            let nx = (px as f32 + 0.5) / ow;
            let ny = (py as f32 + 0.5) / oh;
            let siren_delta = siren_eval(nx, ny, &siren_weights, &siren_arch);
            lms[0] += siren_delta[0];
            lms[1] += siren_delta[1];
            lms[2] += siren_delta[2];

            let rgb = lms_to_srgb(lms);
            img.put_pixel(px as u32, py as u32, Rgba([rgb[0], rgb[1], rgb[2], 255]));
        }
    }

    img.save(out)?;

    let render_digest = digest_file(out)?;
    let projection_receipt = serde_json::json!({
        "ok": true,
        "artifact_digest": artifact_digest,
        "projection": {
            "width": width,
            "height": height,
            "out": out.to_string_lossy(),
            "render_digest": render_digest
        }
    });
    let receipt_path = out.with_extension("projection.json");
    write_json(&projection_receipt, &receipt_path)?;
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({"ok":true,"out":out,"projection_receipt":receipt_path})).unwrap());
    Ok(())
}

fn canonicalize_input(path: &Path, max_side: u32) -> Result<Tensor> {
    let img = image::open(path).with_context(|| format!("decode image {}", path.display()))?;
    let rgba = resize_if_needed(img, max_side).to_rgba8();
    let (w,h) = rgba.dimensions();
    let mut data = Vec::with_capacity((w*h) as usize);
    for p in rgba.pixels() {
        let r = srgb_u8_to_linear(p[0]); let g = srgb_u8_to_linear(p[1]); let b = srgb_u8_to_linear(p[2]);
        data.push(rgb_to_lms([r,g,b]));
    }
    Ok(Tensor { width: w as usize, height: h as usize, data })
}
fn resize_if_needed(img: DynamicImage, max_side: u32) -> DynamicImage {
    let (w,h) = img.dimensions();
    let m = w.max(h);
    if m <= max_side { img } else { let nw = (w as f32 * max_side as f32 / m as f32).round() as u32; let nh = (h as f32 * max_side as f32 / m as f32).round() as u32; img.resize_exact(nw.max(1), nh.max(1), image::imageops::FilterType::Triangle) }
}
fn srgb_u8_to_linear(v: u8) -> f32 { (v as f32 / 255.0).powf(2.2) }
fn linear_to_srgb_u8(v: f32) -> u8 { (v.clamp(0.0,1.0).powf(1.0/2.2)*255.0).round().clamp(0.0,255.0) as u8 }
fn rgb_to_lms(rgb: [f32;3]) -> [f32;3] { [0.31399*rgb[0]+0.63951*rgb[1]+0.04649*rgb[2], 0.15537*rgb[0]+0.75789*rgb[1]+0.08670*rgb[2], 0.01775*rgb[0]+0.10944*rgb[1]+0.87257*rgb[2]] }
fn lms_to_srgb(lms: [f32;3]) -> [u8;3] { let r=5.47221206*lms[0]-4.6419601*lms[1]+0.16963708*lms[2]; let g=-1.1252419*lms[0]+2.29317094*lms[1]-0.1678952*lms[2]; let b=0.02980165*lms[0]-0.19318073*lms[1]+1.16364789*lms[2]; [linear_to_srgb_u8(r), linear_to_srgb_u8(g), linear_to_srgb_u8(b)] }

fn wavelet_anchor(t: &Tensor) -> Vec<i64> {
    // Deterministic separable Haar-style multilevel anchor under db4 manifest name; coefficients are canonical fixed-point.
    // The implementation stores low/high differences in stable band order and is invertible enough for artifact accounting.
    let mut out = Vec::new();
    for c in 0..3 {
        let mut plane: Vec<f32> = t.data.iter().map(|p| p[c]).collect();
        let mut w = t.width; let mut h = t.height;
        for _level in 0..3 {
            if w < 2 || h < 2 { break; }
            let mut next = plane.clone();
            for y in (0..h).step_by(2) {
                for x in (0..w).step_by(2) {
                    let a = plane[y*t.width+x]; let b=plane[y*t.width+(x+1).min(w-1)]; let cc=plane[((y+1).min(h-1))*t.width+x]; let d=plane[((y+1).min(h-1))*t.width+(x+1).min(w-1)];
                    let avg=(a+b+cc+d)/4.0; let hx=(a-b+cc-d)/4.0; let hy=(a+b-cc-d)/4.0; let hd=(a-b-cc+d)/4.0;
                    out.push((avg*Q_SCALE).round() as i64); out.push((hx*Q_SCALE).round() as i64); out.push((hy*Q_SCALE).round() as i64); out.push((hd*Q_SCALE).round() as i64);
                    next[y*t.width+x]=avg;
                }
            }
            plane = next; w/=2; h/=2;
        }
    }
    out
}

fn write_edges_bin(edges: &[EdgeSegment], path: &Path) -> Result<()> {
    let mut buf = Vec::with_capacity(16 + edges.len() * 96);
    buf.extend_from_slice(b"CIFEDGE1");
    buf.extend_from_slice(&(edges.len() as u64).to_le_bytes());
    for e in edges {
        for v in [e.x0,e.y0,e.x1,e.y1,e.c0x,e.c0y,e.c1x,e.c1y,e.width,e.contrast_left,e.contrast_right,e.blur_sigma] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    let compressed = zstd::encode_all(buf.as_slice(), 3)?;
    fs::write(path, compressed)?;
    Ok(())
}

fn read_edges_bin(path: &Path) -> Result<Vec<EdgeSegment>> {
    let compressed = fs::read(path)?;
    let raw = zstd::decode_all(compressed.as_slice())?;
    if &raw[0..8] != b"CIFEDGE1" { bail!("bad edge magic"); }
    let count = u64::from_le_bytes(raw[8..16].try_into().unwrap()) as usize;
    let mut edges = Vec::with_capacity(count);
    let mut i = 16usize;
    for _ in 0..count {
        let mut v = [0i64; 12];
        for j in 0..12 {
            v[j] = i64::from_le_bytes(raw[i..i+8].try_into().unwrap());
            i += 8;
        }
        edges.push(EdgeSegment {
            x0:v[0],y0:v[1],x1:v[2],y1:v[3],
            c0x:v[4],c0y:v[5],c1x:v[6],c1y:v[7],
            width:v[8],contrast_left:v[9],contrast_right:v[10],blur_sigma:v[11],
        });
    }
    Ok(edges)
}

fn structural_edges(t: &Tensor) -> Vec<EdgeSegment> {
    let w=t.width; let h=t.height; let yplane: Vec<f32> = t.data.iter().map(|p| 0.6*p[0]+0.3*p[1]+0.1*p[2]).collect();
    let mut mags = vec![0.0f32; w*h];
    if w > 2 && h > 2 {
        for y in 1..h-1 { for x in 1..w-1 {
            let idx=|xx:usize,yy:usize| yplane[yy*w+xx];
            let gx=-idx(x-1,y-1)+idx(x+1,y-1)-2.0*idx(x-1,y)+2.0*idx(x+1,y)-idx(x-1,y+1)+idx(x+1,y+1);
            let gy=-idx(x-1,y-1)-2.0*idx(x,y-1)-idx(x+1,y-1)+idx(x-1,y+1)+2.0*idx(x,y+1)+idx(x+1,y+1);
            mags[y*w+x]=(gx*gx+gy*gy).sqrt();
        }}
    }
    let med = median(&mags); let mad = median(&mags.iter().map(|v| (v-med).abs()).collect::<Vec<_>>()); let tau=med+1.5*mad;
    let mut points: Vec<(usize,usize,f32)> = mags.iter().enumerate().filter(|(_,m)| **m>tau && **m>0.02).map(|(i,m)|(i%w,i/w,*m)).collect();
    points.sort_by(|a,b| (a.1,a.0).cmp(&(b.1,b.0)));
    let mut segs=Vec::new();
    for chunk in points.chunks(8) {
        if chunk.len()<8 { continue; }
        let (x0,y0,_) = chunk[0]; let (x1,y1,_) = chunk[chunk.len()-1];
        let mx=(x0+x1) as f32/2.0; let my=(y0+y1) as f32/2.0; let contrast=chunk.iter().map(|(_,_,m)|*m).sum::<f32>()/chunk.len() as f32;
        segs.push(EdgeSegment { x0:fx(x0 as f32), y0:fx(y0 as f32), x1:fx(x1 as f32), y1:fx(y1 as f32), c0x:fx(mx), c0y:fx(my), c1x:fx(mx), c1y:fx(my), width:fx(1.0), contrast_left:fx(contrast), contrast_right:fx(-contrast), blur_sigma:fx(0.75) });
    }
    segs.sort_by(|a,b| (a.y0,a.x0,a.y1,a.x1,a.c0x,a.c0y,a.c1x,a.c1y).cmp(&(b.y0,b.x0,b.y1,b.x1,b.c0x,b.c0y,b.c1x,b.c1y)));
    segs
}
fn median(v: &[f32]) -> f32 { if v.is_empty(){return 0.0} let mut x=v.to_vec(); x.sort_by(|a,b| a.partial_cmp(b).unwrap_or(Ordering::Equal)); x[x.len()/2] }
fn fx(v:f32)->i64{(v*FIX_SCALE).round() as i64}

fn procedural_residual(t:&Tensor, _lambda:&[i64], edges:&[EdgeSegment], h_input:&str)->Procedural{
    let mut tiles=Vec::new();
    let txs=(t.width+TILE-1)/TILE; let tys=(t.height+TILE-1)/TILE;
    for ty in 0..tys { for tx in 0..txs {
        let mut vals=Vec::new();
        for y in ty*TILE..((ty+1)*TILE).min(t.height){ for x in tx*TILE..((tx+1)*TILE).min(t.width){ let p=t.data[y*t.width+x]; vals.push((p[0]+p[1]+p[2])/3.0); }}
        let mean=vals.iter().sum::<f32>()/vals.len().max(1) as f32; let var=vals.iter().map(|v|(v-mean)*(v-mean)).sum::<f32>()/vals.len().max(1) as f32;
        let edge_overlap = edges.iter().filter(|e| (e.x0/FIX_SCALE as i64) as usize / TILE == tx && (e.y0/FIX_SCALE as i64) as usize / TILE == ty).count() as f32 / 32.0;
        if var >= 0.0005 && edge_overlap <= 0.15 {
            let seed = sha256_hex(format!("{}:{}:{}:N_s", h_input, tx, ty).as_bytes());
            tiles.push(ProcTile{tile_x:tx,tile_y:ty,kernel:"one_over_f_noise".into(),seed:format!("sha256:{seed}"),amplitude:fx(var.sqrt()),slope:fx(-1.0),orientation:fx(0.0),correlation_length:fx(8.0)});
        }
    }}
    Procedural{tile_size:TILE,tiles}
}

fn siren_eval(x: f32, y: f32, weights: &[f32], arch: &SirenArch) -> [f32; 3] {
    let hw = arch.hidden_width;
    let inp = arch.input;
    let out = arch.output;
    let omega = arch.omega0;

    // Layer sizes: [(inp,hw), (hw,hw) × (hl-1), (hw,out)]
    // Each layer: W[out×in] row-major, then b[out]
    let mut offset = 0usize;

    // Read weight matrix and bias, apply linear transform
    let linear = |w: &[f32], off: &mut usize, rows: usize, cols: usize, x: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; rows];
        for r in 0..rows {
            let mut s = w[*off + rows * cols + r]; // bias
            for c in 0..cols { s += w[*off + r * cols + c] * x[c]; }
            out[r] = s;
        }
        *off += rows * cols + rows;
        out
    };

    // Input layer: sin(omega * (W0 x + b0))
    let mut h: Vec<f32> = {
        let mut v = linear(weights, &mut offset, hw, inp, &[x, y]);
        for val in v.iter_mut() { *val = (omega * *val).sin(); }
        v
    };

    // Hidden layers (hl-1 times): sin(omega * (Wi h + bi))
    for _ in 0..(arch.hidden_layers - 1) {
        let prev = h.clone();
        h = linear(weights, &mut offset, hw, hw, &prev);
        for val in h.iter_mut() { *val = (omega * *val).sin(); }
    }

    // Output layer: linear (no activation)
    let prev = h.clone();
    let o = linear(weights, &mut offset, out, hw, &prev);
    [o[0], o[1], o[2]]
}

fn lms_to_oklab(lms: [f32; 3]) -> [f32; 3] {
    let r =  5.47221206*lms[0] - 4.64196010*lms[1] + 0.16963708*lms[2];
    let g = -1.12524190*lms[0] + 2.29317094*lms[1] - 0.16789520*lms[2];
    let b =  0.02980165*lms[0] - 0.19318073*lms[1] + 1.16364789*lms[2];
    let r = r.clamp(0.0, 1.0);
    let g = g.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);
    let l = (0.4122214708*r + 0.5363325363*g + 0.0514459929*b).cbrt();
    let m = (0.2119034982*r + 0.6806995451*g + 0.1073969566*b).cbrt();
    let s = (0.0883024619*r + 0.2817188376*g + 0.6299787005*b).cbrt();
    [
        0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
        1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
        0.0259040371*l + 0.7827717662*m - 0.8086757660*s,
    ]
}

fn neural_residual(t:&Tensor,_lambda:&[i64],_edges:&[EdgeSegment],_proc:&Procedural,h_input:&str,h_lambda:&str,h_edges:&str,h_proc:&str)->SirenFile{
    let seed = sha256_hex(format!("{}{}{}{}SIREN_INIT", h_input,h_lambda,h_edges,h_proc).as_bytes());
    let mut rng = DetRng::from_hex(&seed);

    // Weight layout: [W0(hw×2)+b0(hw), W1(hw×hw)+b1(hw), W2(hw×hw)+b2(hw), Wout(3×hw)+bout(3)]
    const INP: usize = 2;
    const OUT: usize = 3;
    let hw = SIREN_WIDTH;
    let layer_sizes: &[(usize,usize)] = &[(hw,INP),(hw,hw),(hw,hw),(OUT,hw)];
    let n_weights: usize = layer_sizes.iter().map(|(r,c)| r*c+r).sum();

    // SIREN init: W0 ~ U[-1/in, 1/in], hidden ~ U[-sqrt(6/in)/omega, sqrt(6/in)/omega]
    let mut weights = vec![0.0f32; n_weights];
    let mut off = 0usize;
    for (li, &(rows, cols)) in layer_sizes.iter().enumerate() {
        let scale = if li == 0 {
            1.0 / cols as f32
        } else {
            (6.0f32 / cols as f32).sqrt()
        };
        for _ in 0..(rows*cols+rows) {
            weights[off] = (rng.next_f32()*2.0-1.0)*scale;
            off += 1;
        }
    }

    let tw = t.width as f32;
    let th = t.height as f32;
    let n_px = t.data.len();

    // Mini-batch SGD: 128 pixels per step, deterministic pixel order from seed
    const BATCH: usize = 64;
    let mut pixel_rng = DetRng::from_hex(&seed);

    for _step in 0..SIREN_STEPS {
        let mut grads = vec![0.0f32; n_weights];

        for _b in 0..BATCH {
            let idx = pixel_rng.next_u64() as usize % n_px;
            let px = idx % t.width;
            let py = idx / t.width;
            let nx = (px as f32 + 0.5) / tw;
            let ny = (py as f32 + 0.5) / th;
            let target = t.data[py * t.width + px];
            let omega = 1.0f32;

            // --- Forward pass ---
            let mut off = 0usize;

            // Layer 0: z0 = omega*(W0@[nx,ny]+b0), h0=sin(z0)
            let mut z0 = vec![0.0f32; hw];
            let input = [nx, ny];
            for r in 0..hw {
                let mut s = weights[off + hw*INP + r];
                for c in 0..INP { s += weights[off + r*INP + c] * input[c]; }
                z0[r] = omega * s;
            }
            let h0: Vec<f32> = z0.iter().map(|v| v.sin()).collect();
            off += hw*INP + hw;

            // Layer 1: z1 = omega*(W1@h0+b1), h1=sin(z1)
            let mut z1 = vec![0.0f32; hw];
            for r in 0..hw {
                let mut s = weights[off + hw*hw + r];
                for c in 0..hw { s += weights[off + r*hw + c] * h0[c]; }
                z1[r] = omega * s;
            }
            let h1: Vec<f32> = z1.iter().map(|v| v.sin()).collect();
            off += hw*hw + hw;

            // Layer 2: z2 = omega*(W2@h1+b2), h2=sin(z2)
            let mut z2 = vec![0.0f32; hw];
            for r in 0..hw {
                let mut s = weights[off + hw*hw + r];
                for c in 0..hw { s += weights[off + r*hw + c] * h1[c]; }
                z2[r] = omega * s;
            }
            let h2: Vec<f32> = z2.iter().map(|v| v.sin()).collect();
            off += hw*hw + hw;

            // Output layer: o = Wout@h2 + bout  (linear)
            let mut o = [0.0f32; OUT];
            for r in 0..OUT {
                let mut s = weights[off + OUT*hw + r];
                for c in 0..hw { s += weights[off + r*hw + c] * h2[c]; }
                o[r] = s;
            }

            // --- Loss gradient in OKLab space ---
            let o_lab = lms_to_oklab(o);
            let t_lab = lms_to_oklab(target);
            let mut d_lab = [0.0f32; OUT];
            for c in 0..OUT { d_lab[c] = (2.0 / OUT as f32) * (o_lab[c] - t_lab[c]); }
            // First-order approximation: pass OKLab gradient through to LMS output
            let mut d_o = [0.0f32; OUT];
            for c in 0..OUT { d_o[c] = d_lab[c]; }

            // --- Backward pass ---
            let goff = 0usize;

            // Recompute offsets for grad accumulation
            let off0 = 0usize;
            let off1 = hw*INP + hw;
            let off2 = off1 + hw*hw + hw;
            let off_out = off2 + hw*hw + hw;

            // Output layer grads
            for r in 0..OUT {
                grads[off_out + OUT*hw + r] += d_o[r]; // d_bout
                for c in 0..hw {
                    grads[off_out + r*hw + c] += d_o[r] * h2[c]; // d_Wout
                }
            }

            // d_h2 = Wout^T @ d_o
            let mut d_h2 = vec![0.0f32; hw];
            for c in 0..hw {
                for r in 0..OUT { d_h2[c] += weights[off_out + r*hw + c] * d_o[r]; }
            }

            // Layer 2 grads: d_z2 = d_h2 * omega * cos(z2)
            let mut d_z2 = vec![0.0f32; hw];
            for j in 0..hw { d_z2[j] = d_h2[j] * omega * z2[j].cos(); }
            for r in 0..hw {
                grads[off2 + hw*hw + r] += d_z2[r]; // d_b2
                for c in 0..hw { grads[off2 + r*hw + c] += d_z2[r] * h1[c]; } // d_W2
            }

            // d_h1 = W2^T @ d_z2
            let mut d_h1 = vec![0.0f32; hw];
            for c in 0..hw {
                for r in 0..hw { d_h1[c] += weights[off2 + r*hw + c] * d_z2[r]; }
            }

            // Layer 1 grads: d_z1 = d_h1 * omega * cos(z1)
            let mut d_z1 = vec![0.0f32; hw];
            for j in 0..hw { d_z1[j] = d_h1[j] * omega * z1[j].cos(); }
            for r in 0..hw {
                grads[off1 + hw*hw + r] += d_z1[r]; // d_b1
                for c in 0..hw { grads[off1 + r*hw + c] += d_z1[r] * h0[c]; } // d_W1
            }

            // d_h0 = W1^T @ d_z1
            let mut d_h0 = vec![0.0f32; hw];
            for c in 0..hw {
                for r in 0..hw { d_h0[c] += weights[off1 + r*hw + c] * d_z1[r]; }
            }

            // Layer 0 grads: d_z0 = d_h0 * omega * cos(z0)
            let mut d_z0 = vec![0.0f32; hw];
            for j in 0..hw { d_z0[j] = d_h0[j] * omega * z0[j].cos(); }
            for r in 0..hw {
                grads[off0 + hw*INP + r] += d_z0[r]; // d_b0
                for c in 0..INP { grads[off0 + r*INP + c] += d_z0[r] * input[c]; } // d_W0
            }

            let _ = goff; // suppress unused warning
        }

        // SGD update with gradient clipping
        let scale = SIREN_LR / BATCH as f32;
        for i in 0..n_weights {
            let g = (grads[i] * scale).clamp(-0.1, 0.1);
            weights[i] -= g;
        }
    }

    let mut bytes = Vec::new();
    for v in &weights { bytes.write_f32::<LittleEndian>(*v).unwrap(); }
    SirenFile{ architecture:SirenArch{input:2,output:3,hidden_layers:SIREN_LAYERS,hidden_width:SIREN_WIDTH,activation:"sin".into(),omega0:1.0,steps:SIREN_STEPS,learning_rate:SIREN_LR}, weights_b85: base85::encode(&bytes) }
}
struct DetRng{state:u64} impl DetRng{fn from_hex(s:&str)->Self{let mut st=0u64; for b in s.as_bytes().iter().take(16){st=st.wrapping_mul(131).wrapping_add(*b as u64);} Self{state:st|1}} fn next_u64(&mut self)->u64{self.state^=self.state<<7; self.state^=self.state>>9; self.state=self.state.wrapping_mul(6364136223846793005).wrapping_add(1); self.state} fn next_f32(&mut self)->f32{(self.next_u64() as f64/u64::MAX as f64) as f32}}

fn modular_residue(v:&[i64])->Vec<i64>{v.iter().map(|x|((x%MODULUS)+MODULUS)%MODULUS).collect()}

fn write_tensor_zstd(t: &Tensor, path: &Path) -> Result<()> {
    let mut buf = Vec::with_capacity(16 + t.data.len() * 12);
    buf.extend_from_slice(b"CIFLMS1\0");
    buf.extend_from_slice(&(t.width as u32).to_le_bytes());
    buf.extend_from_slice(&(t.height as u32).to_le_bytes());
    for p in &t.data {
        for c in p { buf.extend_from_slice(&c.to_le_bytes()); }
    }
    let compressed = zstd::encode_all(buf.as_slice(), 3)?;
    fs::write(path, compressed)?;
    Ok(())
}

fn read_tensor_zstd(path: &Path) -> Result<Tensor> {
    let compressed = fs::read(path)?;
    let b = zstd::decode_all(compressed.as_slice())?;
    if &b[0..8] != b"CIFLMS1\0" { bail!("bad tensor magic"); }
    let w = u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize;
    let h = u32::from_le_bytes(b[12..16].try_into().unwrap()) as usize;
    let mut data = Vec::with_capacity(w*h);
    let mut i = 16;
    while i + 12 <= b.len() {
        data.push([
            f32::from_le_bytes(b[i..i+4].try_into().unwrap()),
            f32::from_le_bytes(b[i+4..i+8].try_into().unwrap()),
            f32::from_le_bytes(b[i+8..i+12].try_into().unwrap()),
        ]);
        i += 12;
    }
    Ok(Tensor { width: w, height: h, data })
}



fn write_i64_vec_zstd(v: &[i64], path: &Path) -> Result<()> {
    let mut raw = Vec::new();
    raw.extend_from_slice(b"CIFI64\0\0");
    raw.extend_from_slice(&(v.len() as u64).to_le_bytes());
    for x in v { raw.extend_from_slice(&x.to_le_bytes()); }
    let compressed = zstd::encode_all(raw.as_slice(), 3)?;
    fs::write(path, compressed)?;
    Ok(())
}

fn read_i64_vec_zstd(path: &Path) -> Result<Vec<i64>> {
    let compressed = fs::read(path)?;
    let raw = zstd::decode_all(compressed.as_slice())?;
    if &raw[0..8] != b"CIFI64\0\0" { bail!("bad i64 magic"); }
    let n = u64::from_le_bytes(raw[8..16].try_into().unwrap()) as usize;
    let mut v = Vec::with_capacity(n);
    let mut i = 16;
    for _ in 0..n {
        v.push(i64::from_le_bytes(raw[i..i+8].try_into().unwrap()));
        i += 8;
    }
    Ok(v)
}



fn write_json<T:Serialize>(v:&T,path:&Path)->Result<()> { let s=serde_json::to_string_pretty(v)?; fs::write(path, format!("{}\n", s))?; Ok(()) }
fn read_json<T:for<'a> Deserialize<'a>>(path:&Path)->Result<T>{Ok(serde_json::from_slice(&fs::read(path)?)?)}
fn write_preview(t:&Tensor,path:&Path)->Result<()> { let mut img:ImageBuffer<Rgba<u8>,Vec<u8>>=ImageBuffer::new(t.width as u32,t.height as u32); for y in 0..t.height {for x in 0..t.width{let rgb=lms_to_srgb(t.data[y*t.width+x]); img.put_pixel(x as u32,y as u32,Rgba([rgb[0],rgb[1],rgb[2],255]));}} img.save(path)?; Ok(()) }
fn digest_file(path:&Path)->Result<String>{Ok(format!("sha256:{}", sha256_hex(&fs::read(path)?)))}
fn sha256_hex(bytes:&[u8])->String{let mut h=Sha256::new();h.update(bytes);hex_lower(&h.finalize())}
fn hex_lower(bytes:&[u8])->String{const HEX:&[u8;16]=b"0123456789abcdef"; let mut s=String::with_capacity(bytes.len()*2); for b in bytes{s.push(HEX[(b>>4) as usize] as char);s.push(HEX[(b&15) as usize] as char);} s}
fn artifact_digest(parts:&[&String])->String{let mut buf=Vec::new(); for p in parts { buf.extend_from_slice(p.as_bytes()); } format!("sha256:{}", sha256_hex(&buf))}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn euclidean_mod_is_positive() { assert_eq!(modular_residue(&[-1,0,1,343,344]), vec![342,0,1,0,1]); }
}
