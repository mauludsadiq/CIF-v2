//! edge_tile encoder.
//!
//! Stores a background LMS constant plus up to 4 edge segments.
//! Edges are detected via Sobel on the tile's luminance channel.
//! Payload: 13 + n*24 bytes (n = 0..4).

use crate::rdo::types::{EncodedRegion, LmsTile, RegionEncoder, TILE_SIZE, FIX_SCALE};

const MAX_EDGES: usize = 4;
const EDGE_SCALE: i32 = 1_000_000;

#[derive(Clone)]
struct TileEdge {
    x0: i32, y0: i32, x1: i32, y1: i32,
    contrast: i32,
    sigma: i32,
}

pub struct EdgeTile;

/// Compute luminance from fixed-point LMS.
#[inline]
fn luminance(l: i64, m: i64, s: i64) -> f32 {
    0.6 * l as f32 + 0.3 * m as f32 + 0.1 * s as f32
}

/// Detect up to MAX_EDGES dominant edge segments in a tile via Sobel.
fn detect_edges(tile: &LmsTile) -> Vec<TileEdge> {
    let w = TILE_SIZE;
    let h = TILE_SIZE;

    // Compute luminance plane
    let mut lum = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            lum[y*w+x] = luminance(tile.get(x,y,0), tile.get(x,y,1), tile.get(x,y,2))
                as f32 / FIX_SCALE as f32;
        }
    }

    // Sobel gradient magnitude
    let mut mags = vec![0f32; w * h];
    let mut dirs = vec![(0f32, 0f32); w * h]; // (gx, gy)
    for y in 1..h-1 {
        for x in 1..w-1 {
            let idx = |xx: usize, yy: usize| lum[yy*w+xx];
            let gx = -idx(x-1,y-1) + idx(x+1,y-1)
                     - 2.0*idx(x-1,y) + 2.0*idx(x+1,y)
                     - idx(x-1,y+1) + idx(x+1,y+1);
            let gy = -idx(x-1,y-1) - 2.0*idx(x,y-1) - idx(x+1,y-1)
                     + idx(x-1,y+1) + 2.0*idx(x,y+1) + idx(x+1,y+1);
            let mag = (gx*gx + gy*gy).sqrt();
            mags[y*w+x] = mag;
            dirs[y*w+x] = (gx, gy);
        }
    }

    // Find threshold (median + 1.5*MAD)
    let mut sorted = mags.clone();
    sorted.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let med = sorted[sorted.len()/2];
    let mut diffs: Vec<f32> = sorted.iter().map(|v| (v-med).abs()).collect();
    diffs.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let mad = diffs[diffs.len()/2];
    let tau = (med + 1.5*mad).max(0.05);

    // Collect strong edge pixels, sort by magnitude descending
    let mut points: Vec<(usize, usize, f32)> = mags.iter().enumerate()
        .filter(|(_,&m)| m > tau)
        .map(|(i,&m)| (i%w, i/w, m))
        .collect();
    points.sort_by(|a,b| b.2.partial_cmp(&a.2).unwrap());

    // Group into up to MAX_EDGES segments by taking top clusters
    let mut edges = Vec::new();
    let mut used = vec![false; w*h];

    for &(px, py, pmag) in &points {
        if edges.len() >= MAX_EDGES { break; }
        if used[py*w+px] { continue; }

        // Collect nearby pixels (within 4px) as one segment
        let mut seg_pts: Vec<(usize, usize, f32)> = vec![(px, py, pmag)];
        used[py*w+px] = true;

        for &(qx, qy, qmag) in &points {
            if used[qy*w+qx] { continue; }
            let dx = (px as i32 - qx as i32).abs();
            let dy = (py as i32 - qy as i32).abs();
            if dx <= 4 && dy <= 4 {
                seg_pts.push((qx, qy, qmag));
                used[qy*w+qx] = true;
            }
        }

        if seg_pts.len() < 2 { continue; }

        // Fit segment endpoints as bbox extremes
        let x0 = seg_pts.iter().map(|p| p.0).min().unwrap();
        let y0 = seg_pts.iter().map(|p| p.1).min().unwrap();
        let x1 = seg_pts.iter().map(|p| p.0).max().unwrap();
        let y1 = seg_pts.iter().map(|p| p.1).max().unwrap();
        let contrast = seg_pts.iter().map(|p| p.2).sum::<f32>() / seg_pts.len() as f32;

        edges.push(TileEdge {
            x0: (x0 as i32 * EDGE_SCALE),
            y0: (y0 as i32 * EDGE_SCALE),
            x1: (x1 as i32 * EDGE_SCALE),
            y1: (y1 as i32 * EDGE_SCALE),
            contrast: (contrast * EDGE_SCALE as f32) as i32,
            sigma: (EDGE_SCALE * 3 / 4), // 0.75 canonical pixels
        });
    }

    edges
}

/// Rasterize edges onto a tile (same logic as CIF v2 compositor).
fn rasterize_edges(tile: &mut LmsTile, bg: [i64;3], edges: &[TileEdge]) {
    let scale = EDGE_SCALE as f32;
    for y in 0..TILE_SIZE {
        for x in 0..TILE_SIZE {
            let mut lms = bg;
            let cx = x as f32 + 0.5;
            let cy = y as f32 + 0.5;

            for e in edges {
                let p0x = e.x0 as f32 / scale;
                let p0y = e.y0 as f32 / scale;
                let p1x = e.x1 as f32 / scale;
                let p1y = e.y1 as f32 / scale;
                let dx = p1x - p0x; let dy = p1y - p0y;
                let len2 = dx*dx + dy*dy;
                let contrast = e.contrast as f32 / scale;
                let sigma = e.sigma as f32 / scale;
                let half_w = 0.5f32;

                let (dist, signed) = if len2 < 1e-6 {
                    (((cx-p0x).powi(2)+(cy-p0y).powi(2)).sqrt(), 1.0f32)
                } else {
                    let t = ((cx-p0x)*dx+(cy-p0y)*dy)/len2;
                    let t = t.clamp(0.0,1.0);
                    let nx = cx-(p0x+t*dx); let ny = cy-(p0y+t*dy);
                    let dist = (nx*nx+ny*ny).sqrt();
                    (dist, (dx*ny-dy*nx).signum())
                };

                let outer = half_w + sigma;
                if dist < outer {
                    let tb = ((outer-dist)/sigma.max(1e-6)).clamp(0.0,1.0);
                    let smooth = tb*tb*(3.0-2.0*tb);
                    let inf = (contrast * smooth * signed * FIX_SCALE as f32) as i64;
                    lms[0] = (lms[0] + inf*6/10).clamp(0, 2*FIX_SCALE);
                    lms[1] = (lms[1] + inf*3/10).clamp(0, 2*FIX_SCALE);
                    lms[2] = (lms[2] + inf*1/10).clamp(0, 2*FIX_SCALE);
                }
            }

            tile.set(x, y, 0, lms[0]);
            tile.set(x, y, 1, lms[1]);
            tile.set(x, y, 2, lms[2]);
        }
    }
}

impl RegionEncoder for EdgeTile {
    fn name(&self) -> &'static str { "edge_tile" }

    fn encode(&self, tile: &LmsTile) -> EncodedRegion {
        // Detect edges first
        let edges = detect_edges(tile);

        // Compute background: if edges exist, use mean of pixels far from any edge.
        // Otherwise use full tile mean.
        let bg = if edges.is_empty() {
            let n = (TILE_SIZE * TILE_SIZE) as i64;
            let mut sum = [0i64; 3];
            for y in 0..TILE_SIZE {
                for x in 0..TILE_SIZE {
                    for c in 0..3 { sum[c] += tile.get(x, y, c); }
                }
            }
            [sum[0]/n, sum[1]/n, sum[2]/n]
        } else {
            // Use pixels more than 4 canonical pixels from any edge segment
            let scale = EDGE_SCALE as f32;
            let mut sum = [0i64; 3];
            let mut count = 0i64;
            for y in 0..TILE_SIZE {
                for x in 0..TILE_SIZE {
                    let cx = x as f32 + 0.5;
                    let cy = y as f32 + 0.5;
                    let mut min_dist = f32::MAX;
                    for e in &edges {
                        let p0x = e.x0 as f32 / scale;
                        let p0y = e.y0 as f32 / scale;
                        let p1x = e.x1 as f32 / scale;
                        let p1y = e.y1 as f32 / scale;
                        let dx = p1x - p0x; let dy = p1y - p0y;
                        let len2 = dx*dx + dy*dy;
                        let dist = if len2 < 1e-6 {
                            ((cx-p0x).powi(2)+(cy-p0y).powi(2)).sqrt()
                        } else {
                            let t = ((cx-p0x)*dx+(cy-p0y)*dy)/len2;
                            let t = t.clamp(0.0,1.0);
                            let nx=cx-(p0x+t*dx); let ny=cy-(p0y+t*dy);
                            (nx*nx+ny*ny).sqrt()
                        };
                        if dist < min_dist { min_dist = dist; }
                    }
                    if min_dist > 4.0 {
                        for c in 0..3 { sum[c] += tile.get(x, y, c); }
                        count += 1;
                    }
                }
            }
            if count > 0 {
                [sum[0]/count, sum[1]/count, sum[2]/count]
            } else {
                // All pixels near edges — fall back to mean
                let n = (TILE_SIZE * TILE_SIZE) as i64;
                let mut s = [0i64; 3];
                for y in 0..TILE_SIZE {
                    for x in 0..TILE_SIZE {
                        for c in 0..3 { s[c] += tile.get(x, y, c); }
                    }
                }
                [s[0]/n, s[1]/n, s[2]/n]
            }
        };

        // Serialize payload
        let mut payload = Vec::new();
        for c in 0..3 {
            payload.extend_from_slice(&(bg[c] as i32).to_le_bytes());
        }
        payload.push(edges.len() as u8);
        for e in &edges {
            payload.extend_from_slice(&e.x0.to_le_bytes());
            payload.extend_from_slice(&e.y0.to_le_bytes());
            payload.extend_from_slice(&e.x1.to_le_bytes());
            payload.extend_from_slice(&e.y1.to_le_bytes());
            payload.extend_from_slice(&e.contrast.to_le_bytes());
            payload.extend_from_slice(&e.sigma.to_le_bytes());
        }

        let rate_bits = (payload.len() * 8) as u64;
        EncodedRegion {
            encoder_name: self.name(),
            rate_bits,
            decode_cost: (TILE_SIZE * TILE_SIZE * 3 * (1 + edges.len())) as u64,
            encode_cost: (TILE_SIZE * TILE_SIZE * 3 * 10) as u64,
            memory_cost: payload.len() as u64,
            replay_cost: (TILE_SIZE * TILE_SIZE * 3 * 10) as u64,
            payload,
        }
    }

    fn decode(&self, encoded: &EncodedRegion, out: &mut LmsTile) {
        let p = &encoded.payload;
        let bg = [
            i32::from_le_bytes(p[0..4].try_into().unwrap()) as i64,
            i32::from_le_bytes(p[4..8].try_into().unwrap()) as i64,
            i32::from_le_bytes(p[8..12].try_into().unwrap()) as i64,
        ];
        let n_edges = p[12] as usize;
        let mut edges = Vec::new();
        for i in 0..n_edges {
            let base = 13 + i * 24;
            edges.push(TileEdge {
                x0: i32::from_le_bytes(p[base..base+4].try_into().unwrap()),
                y0: i32::from_le_bytes(p[base+4..base+8].try_into().unwrap()),
                x1: i32::from_le_bytes(p[base+8..base+12].try_into().unwrap()),
                y1: i32::from_le_bytes(p[base+12..base+16].try_into().unwrap()),
                contrast: i32::from_le_bytes(p[base+16..base+20].try_into().unwrap()),
                sigma: i32::from_le_bytes(p[base+20..base+24].try_into().unwrap()),
            });
        }
        rasterize_edges(out, bg, &edges);
    }

    fn rate_bits(&self, e: &EncodedRegion) -> u64 { e.rate_bits }
    fn decode_cost(&self, e: &EncodedRegion) -> u64 { e.decode_cost }
    fn encode_cost(&self, e: &EncodedRegion) -> u64 { e.encode_cost }
    fn memory_cost(&self, e: &EncodedRegion) -> u64 { e.memory_cost }
    fn replay_cost(&self, e: &EncodedRegion) -> u64 { e.replay_cost }
}
