//! Compact binary tree format for CIF-RDO v0.1
//!
//! Replaces verbose tree.json with a fixed-header + per-tile record format.
//!
//! Layout:
//!   [0..8]   magic "CIFRDOT1"
//!   [8..10]  tile_size: u16 le
//!   [10..12] grid_w: u16 le
//!   [12..14] grid_h: u16 le
//!   [14..18] quality_lambda: u32 le
//!   [18..20] visible_w: u16 le  (wait — use u32 for large images)
//!
//! Actually use u32 for image dimensions:
//!   [0..8]   magic "CIFRDOT1"
//!   [8..10]  tile_size: u16 le
//!   [10..12] grid_w: u16 le
//!   [12..14] grid_h: u16 le
//!   [14..18] visible_w: u32 le
//!   [18..22] visible_h: u32 le
//!   [22..26] quality_lambda: u32 le
//!   [26..]   per-tile records: (encoder_id: u8, payload_len: u32 le) × n_tiles

use anyhow::{bail, Result};

pub const MAGIC: &[u8; 8] = b"CIFRDOT1";

pub const ENCODER_IDS: &[&str] = &[
    "constant_lms",
    "affine_lms",
    "quadratic_lms",
    "wavelet_tile",
    "edge_tile",
    "micro_siren_tile",
    "dct_tile",
];

pub fn encoder_id(name: &str) -> u8 {
    ENCODER_IDS.iter().position(|&n| n == name).unwrap_or(0) as u8
}

pub fn encoder_name(id: u8) -> &'static str {
    ENCODER_IDS.get(id as usize).copied().unwrap_or("constant_lms")
}

pub struct TreeHeader {
    pub tile_size: u16,
    pub grid_w: u16,
    pub grid_h: u16,
    pub visible_w: u32,
    pub visible_h: u32,
    pub quality_lambda: u32,
}

pub struct TileRecord {
    pub encoder_id: u8,
    pub payload_len: u32,
}

/// Serialize tree to compact binary format.
pub fn write_tree(header: &TreeHeader, tiles: &[TileRecord]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(26 + tiles.len() * 5);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&header.tile_size.to_le_bytes());
    buf.extend_from_slice(&header.grid_w.to_le_bytes());
    buf.extend_from_slice(&header.grid_h.to_le_bytes());
    buf.extend_from_slice(&header.visible_w.to_le_bytes());
    buf.extend_from_slice(&header.visible_h.to_le_bytes());
    buf.extend_from_slice(&header.quality_lambda.to_le_bytes());
    for t in tiles {
        buf.push(t.encoder_id);
        buf.extend_from_slice(&t.payload_len.to_le_bytes());
    }
    buf
}

/// Parse compact binary tree format.
pub fn read_tree(data: &[u8]) -> Result<(TreeHeader, Vec<TileRecord>)> {
    if data.len() < 26 { bail!("tree.bin too short"); }
    if &data[0..8] != MAGIC { bail!("bad tree.bin magic"); }
    let tile_size    = u16::from_le_bytes(data[8..10].try_into().unwrap());
    let grid_w       = u16::from_le_bytes(data[10..12].try_into().unwrap());
    let grid_h       = u16::from_le_bytes(data[12..14].try_into().unwrap());
    let visible_w    = u32::from_le_bytes(data[14..18].try_into().unwrap());
    let visible_h    = u32::from_le_bytes(data[18..22].try_into().unwrap());
    let quality_lambda = u32::from_le_bytes(data[22..26].try_into().unwrap());
    let header = TreeHeader { tile_size, grid_w, grid_h, visible_w, visible_h, quality_lambda };

    let n = (grid_w as usize) * (grid_h as usize);
    let mut tiles = Vec::with_capacity(n);
    let mut pos = 26usize;
    for _ in 0..n {
        if pos + 5 > data.len() { bail!("tree.bin truncated"); }
        let encoder_id = data[pos];
        let payload_len = u32::from_le_bytes(data[pos+1..pos+5].try_into().unwrap());
        tiles.push(TileRecord { encoder_id, payload_len });
        pos += 5;
    }
    Ok((header, tiles))
}
