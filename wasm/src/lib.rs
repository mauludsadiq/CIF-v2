use wasm_bindgen::prelude::*;
use std::io::Read;

const CIFV2_MAGIC: &[u8; 8] = b"CIFV2\x00\x00\x00";
const SCALE: f32 = 1_000_000.0;

#[derive(Clone)]
struct Tensor { width: usize, height: usize, data: Vec<[f32;3]> }

struct EdgeSegment {
    x0:i64,y0:i64,x1:i64,y1:i64,
    c0x:i64,c0y:i64,c1x:i64,c1y:i64,
    width:i64,contrast_left:i64,contrast_right:i64,blur_sigma:i64,
}

struct SirenWeights {
    weights: Vec<f32>,
    hidden_width: usize,
    hidden_layers: usize,
    omega0: f32,
}

fn zstd_decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    let cursor = std::io::Cursor::new(data);
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(cursor)
        .map_err(|e| format!("zstd init: {e}"))?;
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).map_err(|e| format!("zstd decode: {e}"))?;
    Ok(out)
}

fn read_tensor(data: &[u8]) -> Result<Tensor, String> {
    let raw = zstd_decompress(data)?;
    if &raw[0..8] != b"CIFLMS1\x00" { return Err("bad tensor magic".into()); }
    let w = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
    let h = u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize;
    let mut pixels = Vec::with_capacity(w*h);
    let mut i = 16;
    while i+12 <= raw.len() {
        pixels.push([
            f32::from_le_bytes(raw[i..i+4].try_into().unwrap()),
            f32::from_le_bytes(raw[i+4..i+8].try_into().unwrap()),
            f32::from_le_bytes(raw[i+8..i+12].try_into().unwrap()),
        ]);
        i += 12;
    }
    Ok(Tensor { width: w, height: h, data: pixels })
}

fn read_edges(data: &[u8]) -> Result<Vec<EdgeSegment>, String> {
    let raw = zstd_decompress(data)?;
    if &raw[0..8] != b"CIFEDGE1" { return Err("bad edge magic".into()); }
    let count = u64::from_le_bytes(raw[8..16].try_into().unwrap()) as usize;
    let mut edges = Vec::with_capacity(count);
    let mut i = 16usize;
    for _ in 0..count {
        let mut v = [0i64;12];
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

fn read_siren(data: &[u8]) -> Result<SirenWeights, String> {
    let json: serde_json::Value = serde_json::from_slice(data)
        .map_err(|e| format!("siren json: {e}"))?;
    let arch = &json["architecture"];
    let hw = arch["hidden_width"].as_u64().unwrap_or(32) as usize;
    let hl = arch["hidden_layers"].as_u64().unwrap_or(3) as usize;
    let omega0 = arch["omega0"].as_f64().unwrap_or(1.0) as f32;
    let b85 = json["weights_b85"].as_str().ok_or("missing weights")?;
    let raw = base85::decode(b85).map_err(|e| format!("b85: {e}"))?;
    let weights: Vec<f32> = raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]]))
        .collect();
    Ok(SirenWeights { weights, hidden_width: hw, hidden_layers: hl, omega0 })
}

fn siren_eval(x: f32, y: f32, s: &SirenWeights) -> [f32;3] {
    let hw = s.hidden_width;
    let omega = s.omega0;
    let mut off = 0usize;
    let linear = |w: &[f32], off: &mut usize, rows: usize, cols: usize, input: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; rows];
        for r in 0..rows {
            let mut s = w[*off + rows*cols + r];
            for c in 0..cols { s += w[*off + r*cols + c] * input[c]; }
            out[r] = s;
        }
        *off += rows*cols + rows;
        out
    };
    let mut h = linear(&s.weights, &mut off, hw, 2, &[x,y]);
    for v in h.iter_mut() { *v = (omega * *v).sin(); }
    for _ in 0..(s.hidden_layers-1) {
        let prev = h.clone();
        h = linear(&s.weights, &mut off, hw, hw, &prev);
        for v in h.iter_mut() { *v = (omega * *v).sin(); }
    }
    let prev = h.clone();
    let o = linear(&s.weights, &mut off, 3, hw, &prev);
    [o[0], o[1], o[2]]
}

fn lms_to_srgb(lms: [f32;3]) -> [u8;3] {
    let r = (5.47221206*lms[0]-4.6419601*lms[1]+0.16963708*lms[2]).clamp(0.0,1.0);
    let g = (-1.1252419*lms[0]+2.29317094*lms[1]-0.1678952*lms[2]).clamp(0.0,1.0);
    let b = (0.02980165*lms[0]-0.19318073*lms[1]+1.16364789*lms[2]).clamp(0.0,1.0);
    let f = |v: f32| (v.powf(1.0/2.2)*255.0).round().clamp(0.0,255.0) as u8;
    [f(r), f(g), f(b)]
}

fn render_frame(t: &Tensor, edges: &[EdgeSegment], siren: &SirenWeights,
                width: usize, height: usize) -> Vec<u8> {
    let cw = t.width as f32;
    let ch = t.height as f32;
    let ow = width as f32;
    let oh = height as f32;
    let mag = (ow/cw).max(oh/ch).max(1.0);
    let edge_weight = (1.0 - 1.0/mag).clamp(0.0, 1.0);
    let siren_weight = edge_weight;

    const TILE: usize = 32;
    let tiles_x = (width + TILE - 1) / TILE;
    let tiles_y = (height + TILE - 1) / TILE;
    let mut tile_edges: Vec<Vec<usize>> = vec![Vec::new(); tiles_x*tiles_y];

    for (ei, seg) in edges.iter().enumerate() {
        let pad = (seg.width + seg.blur_sigma) as f32 / SCALE;
        let ex0 = seg.x0.min(seg.x1).min(seg.c0x).min(seg.c1x) as f32/SCALE - pad;
        let ex1 = seg.x0.max(seg.x1).max(seg.c0x).max(seg.c1x) as f32/SCALE + pad;
        let ey0 = seg.y0.min(seg.y1).min(seg.c0y).min(seg.c1y) as f32/SCALE - pad;
        let ey1 = seg.y0.max(seg.y1).max(seg.c0y).max(seg.c1y) as f32/SCALE + pad;
        let px0 = ((ex0*ow/cw) as usize).saturating_sub(1);
        let px1 = ((ex1*ow/cw) as usize+1).min(width-1);
        let py0 = ((ey0*oh/ch) as usize).saturating_sub(1);
        let py1 = ((ey1*oh/ch) as usize+1).min(height-1);
        for ty in (py0/TILE)..=(py1/TILE).min(tiles_y-1) {
            for tx in (px0/TILE)..=(px1/TILE).min(tiles_x-1) {
                tile_edges[ty*tiles_x+tx].push(ei);
            }
        }
    }

    let mut out = vec![0u8; width*height*4];
    for py in 0..height {
        let ty = py/TILE;
        for px in 0..width {
            let tx = px/TILE;
            let cx = (px as f32+0.5)*cw/ow;
            let cy = (py as f32+0.5)*ch/oh;
            let fx = cx-0.5;
            let fy = cy-0.5;
            let x0 = (fx.floor() as isize).clamp(0,t.width as isize-1) as usize;
            let y0 = (fy.floor() as isize).clamp(0,t.height as isize-1) as usize;
            let x1 = (x0+1).min(t.width-1);
            let y1 = (y0+1).min(t.height-1);
            let tx_ = (fx-fx.floor()).clamp(0.0,1.0);
            let ty_ = (fy-fy.floor()).clamp(0.0,1.0);
            let s = |xi,yi| t.data[yi*t.width+xi];
            let lerp3 = |a:[f32;3],b:[f32;3],t:f32| -> [f32;3] {
                [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t]
            };
            let mut lms = lerp3(lerp3(s(x0,y0),s(x1,y0),tx_),
                                lerp3(s(x0,y1),s(x1,y1),tx_),ty_);

            // Edge compositor disabled in WASM viewer — base tensor + SIREN only
            let _ = &tile_edges;
            let _ = edge_weight;

            // SIREN contribution disabled — base tensor render only
            let _ = siren;
            let _ = siren_weight;

            // Clamp LMS before conversion to prevent color overflow
            lms[0]=lms[0].clamp(0.0,1.0);
            lms[1]=lms[1].clamp(0.0,1.0);
            lms[2]=lms[2].clamp(0.0,1.0);
            let rgb=lms_to_srgb(lms);
            let i=(py*width+px)*4;
            out[i]=rgb[0]; out[i+1]=rgb[1]; out[i+2]=rgb[2]; out[i+3]=255;
        }
    }
    out
}

// --- Public WASM API ---

#[wasm_bindgen]
pub struct CifArtifact {
    tensor: Tensor,
    edges: Vec<EdgeSegment>,
    siren: SirenWeights,
    artifact_digest: String,
}

#[wasm_bindgen]
impl CifArtifact {
    #[wasm_bindgen(getter)]
    pub fn canonical_width(&self) -> u32 { self.tensor.width as u32 }

    #[wasm_bindgen(getter)]
    pub fn canonical_height(&self) -> u32 { self.tensor.height as u32 }

    #[wasm_bindgen(getter)]
    pub fn artifact_digest(&self) -> String { self.artifact_digest.clone() }

    pub fn render(&self, width: u32, height: u32) -> Vec<u8> {
        render_frame(&self.tensor, &self.edges, &self.siren,
                     width as usize, height as usize)
    }
}

#[wasm_bindgen]
pub fn load_cifv2f(data: &[u8]) -> Result<CifArtifact, String> {
    if data.len() < 16 { return Err("file too short".into()); }
    if &data[0..8] != CIFV2_MAGIC { return Err("not a .cifv2f file".into()); }
    let n = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;

    let mut components: std::collections::HashMap<String, Vec<u8>> = Default::default();
    let mut pos = 16usize;
    for _ in 0..n {
        let entry = &data[pos..pos+64];
        let name_end = entry[..32].iter().position(|&b| b==0).unwrap_or(32);
        let name = std::str::from_utf8(&entry[..name_end])
            .map_err(|e| format!("name utf8: {e}"))?.to_string();
        let offset = u64::from_le_bytes(entry[32..40].try_into().unwrap()) as usize;
        let length = u64::from_le_bytes(entry[40..48].try_into().unwrap()) as usize;
        components.insert(name, data[offset..offset+length].to_vec());
        pos += 64;
    }

    let tensor = read_tensor(components.get("canonical_lms.tensor")
        .ok_or("missing canonical_lms.tensor")?)?;
    let edges = read_edges(components.get("edges.cifedge")
        .ok_or("missing edges.cifedge")?)?;
    let siren = read_siren(components.get("inr.siren")
        .ok_or("missing inr.siren")?)?;

    // Compute artifact digest from receipt
    let receipt_bytes = components.get("receipt.json")
        .ok_or("missing receipt.json")?;
    let receipt: serde_json::Value = serde_json::from_slice(receipt_bytes)
        .map_err(|e| format!("receipt json: {e}"))?;
    let artifact_digest = receipt["artifact_digest"]
        .as_str().unwrap_or("unknown").to_string();

    Ok(CifArtifact { tensor, edges, siren, artifact_digest })
}
