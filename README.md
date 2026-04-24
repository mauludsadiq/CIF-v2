# CIF v2 — Collapse Image Format v2

CIF v2 is a deterministic Rust implementation of the **Φ-CIFv2 collapse operator**:

```text
raster image → canonical executable visual-field artifact
```

It does not claim that an input raster magically contains unknown physical detail. Instead, it deterministically decomposes a raster measurement into a replay-verifiable artifact with:

1. a canonical LMS measurement tensor,
2. a perceptual wavelet anchor,
3. explicit structural edge segments,
4. deterministic procedural texture descriptors,
5. a compact SIREN residual parameter file,
6. a modular residue fingerprint,
7. a manifest and FARD-compatible receipt chain.

The implementation is intentionally conservative: every output file is produced by executable Rust code, every digest is recomputed by `verify`, and `replay` reruns Φ against the original input to prove deterministic reproducibility.

---

## Repository Layout

```text
CIF_v2/
├── Cargo.toml
├── README.md
└── src/
    └── main.rs
```

The CLI binary is named:

```text
cifv2
```

---

## Build

From the repository root:

```bash
cargo build --release
```

Run tests:

```bash
cargo test
```

---

## CLI

### Encode

```bash
cargo run --release -- encode --input input.png --out capture.cifv2
```

Optional maximum canonical side length:

```bash
cargo run --release -- encode --input input.jpg --out capture.cifv2 --max-side 512
```

The encoder creates:

```text
capture.cifv2/
├── manifest.json
├── preview.png
├── canonical_lms.tensor
├── lambda_real.bin
├── edges.cifedge
├── procedural.json
├── inr.siren
├── lambda_mod.bin
└── receipt.json
```

### Verify

```bash
cargo run --release -- verify --artifact capture.cifv2
```

Verification checks:

1. manifest format and operator,
2. every file digest,
3. modular residue recomputation,
4. manifest digest,
5. final artifact digest.

Expected success shape:

```json
{
  "ok": true,
  "artifact_digest": "sha256:...",
  "steps_verified": 6
}
```

### Replay

```bash
cargo run --release -- replay --input input.png --artifact capture.cifv2 --temp-out replay.cifv2
```

Replay reruns the full collapse operator and compares the new artifact digest against the saved receipt.

### Render Preview Projection

```bash
cargo run --release -- render --artifact capture.cifv2 --out render.png --width 1920 --height 1080
```

This command renders a projection from the canonical LMS measurement tensor. It is a compatibility projection, not a claim that unknown physical detail has been recovered.

---

## Φ-CIFv2 Algorithm

The implemented operator is:

```text
Φ_CIFv2(I):

1. canonicalize_input(I)        → canonical_lms.tensor
2. wavelet_anchor(T0)           → lambda_real.bin
3. structural_edges(T0)         → edges.cifedge
4. procedural_residual(...)     → procedural.json
5. neural_residual(...)         → inr.siren
6. modular_residue(lambda_real) → lambda_mod.bin
7. assemble(...)                → manifest.json
8. receipt(...)                 → receipt.json
```

Mathematically:

```text
Φ(I) = (Wλ, Eβ, Ns, fθ, ρ, R)
```

Where:

| Component | File | Meaning |
|---|---|---|
| canonical LMS tensor | `canonical_lms.tensor` | deterministic linear LMS measurement |
| wavelet anchor | `lambda_real.bin` | fixed-point perceptual coefficients |
| structural edges | `edges.cifedge` | deterministic cubic segment records |
| procedural layer | `procedural.json` | deterministic one-over-f tile descriptors |
| neural residual | `inr.siren` | deterministic SIREN residual parameters |
| residue | `lambda_mod.bin` | λ mod 343 fingerprint |
| receipt | `receipt.json` | replay-verifiable digest chain |

---

## Determinism Rules

The implementation fixes:

- image metadata stripping by canonical decode path,
- canonical max-side resize using Triangle filtering,
- sRGB gamma linearization with γ = 2.2,
- RGB → LMS matrix,
- little-endian binary serialization,
- fixed-point coefficient quantization,
- Euclidean modular projection,
- stable JSON serialization,
- stable digest ordering,
- deterministic seed derivation via SHA-256.

The residue layer is computed as:

```text
λ_mod[k] = ((λ_real[k] % 343) + 343) % 343
```

`lambda_mod.bin` is validation metadata. It is not used to reconstruct the image.

---

## Artifact Contract

### `canonical_lms.tensor`

Binary layout:

```text
magic: 8 bytes = "CIFLMS1\0"
width: u32 little-endian
height: u32 little-endian
pixels: repeated f32 little-endian triples [L, M, S]
```

### `lambda_real.bin`

Binary layout:

```text
magic: 8 bytes = "CIFI64\0\0"
count: u64 little-endian
values: repeated i64 little-endian fixed-point coefficients
```

### `lambda_mod.bin`

Same binary layout as `lambda_real.bin`, but values are in:

```text
Z / 343Z
```

### `edges.cifedge`

Canonical JSON array of edge segment records:

```json
{
  "x0": 0,
  "y0": 0,
  "x1": 1000000,
  "y1": 1000000,
  "c0x": 500000,
  "c0y": 500000,
  "c1x": 500000,
  "c1y": 500000,
  "width": 1000000,
  "contrast_left": 1200,
  "contrast_right": -1200,
  "blur_sigma": 750000
}
```

All numeric fields are fixed-point integers with scale `1_000_000`.

### `procedural.json`

Canonical JSON procedural texture descriptor:

```json
{
  "tile_size": 32,
  "tiles": [
    {
      "tile_x": 0,
      "tile_y": 0,
      "kernel": "one_over_f_noise",
      "seed": "sha256:...",
      "amplitude": 1000,
      "slope": -1000000,
      "orientation": 0,
      "correlation_length": 8000000
    }
  ]
}
```

### `inr.siren`

Canonical JSON containing architecture and Base85-encoded little-endian f32 weights:

```json
{
  "architecture": {
    "input": 2,
    "output": 3,
    "hidden_layers": 3,
    "hidden_width": 32,
    "activation": "sin",
    "omega0": 30.0,
    "steps": 256,
    "learning_rate": 0.0001
  },
  "weights_b85": "..."
}
```

### `manifest.json`

The manifest binds all artifact components by SHA-256 digest.

### `receipt.json`

The receipt is FARD-compatible in structure: a deterministic step chain plus a final artifact digest.

---

## Current Implementation Boundary

This is a real working CIF v2 reference implementation, not a mock. It produces artifacts, receipts, verification results, replay checks, and render projections.

The current reference encoder uses a deterministic multilevel wavelet-like anchor and labels the manifest as the CIF v2 anchor family used by this code path. Future versions can replace the anchor with an exact Daubechies-4 lifting implementation without changing the surrounding artifact contract.

The SIREN layer is deterministic and serialized as executable parameters. The reference fitting path is intentionally CPU-stable and compact; future versions can increase training depth while preserving replay determinism by fixing float mode, optimizer state, and batching.

---

## Development Notes for VS Code

Recommended extensions:

- rust-analyzer
- CodeLLDB
- Even Better TOML

Useful commands:

```bash
cargo fmt
cargo test
cargo run --release -- encode --input input.png --out capture.cifv2
cargo run --release -- verify --artifact capture.cifv2
```

---

## Definition

CIF v2 is not a standard raster codec. It is a deterministic, receipt-verifiable visual artifact system:

```text
measurement anchor
+ structural boundaries
+ procedural texture descriptors
+ neural residual field
+ modular residue fingerprint
+ replay receipt
```

Pixels are outputs of rendering. They are not the identity of the artifact.
