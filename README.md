# CIF v2 — Collapse Image Format v2

CIF v2 encodes a raster image into a deterministic, replay-verifiable artifact that renders at any resolution — from thumbnail to billboard — without degradation.

```
input image  →  CIF artifact  →  render at any scale
```

The artifact is the image. Raster files are projections of it.

-----

## Core Concept

Standard image formats store pixels. CIF v2 stores the image as an executable field:

```
canonical_lms.tensor   perceptual measurement anchor
edges.cifedge          resolution-independent structural boundaries
inr.siren              trained continuous color field
procedural.json        deterministic texture descriptors
lambda_real.bin        fixed-point wavelet coefficients
lambda_mod.bin         modular residue fingerprint
manifest.json          component digest binding
receipt.json           replay-verifiable step chain
```

A PNG or JPEG decoder sees only `preview.png` — a valid thumbnail requiring no CIF awareness. A CIF-aware renderer opens the artifact and renders at any target resolution using the executable layer.

Resolution independence is carried by two components:

- **`edges.cifedge`** — cubic Bézier segments in fixed-point coordinates. No raster grid. Boundaries render at full sharpness at any output resolution.
- **`inr.siren`** — a trained implicit neural field `f(x,y) → (L,M,S)` defined over `[0,1]²`. Continuous. No native resolution. Sampled at the target pixel density at render time.

-----

## Projection Contract

```
artifact_digest   identity of the source artifact — invariant across all renders
render_digest     sha256 of the output PNG — deterministic given (artifact, width, height)
```

Every render call produces a projection receipt:

```json
{
  "ok": true,
  "artifact_digest": "sha256:...",
  "projection": {
    "width": 4096,
    "height": 4096,
    "out": "render.png",
    "render_digest": "sha256:..."
  }
}
```

The artifact digest never changes. The render digest changes with resolution. Pixels are outputs, not identity.

-----

## Build

```bash
cargo build --release
cargo test
```

Binary: `./target/release/cifv2`

-----

## CLI

### Encode

```bash
cifv2 encode --input photo.png --out capture.cifv2
cifv2 encode --input photo.jpg --out capture.cifv2 --max-side 512
```

Encodes the input image into a CIF artifact directory. Runs the full Φ operator including SIREN training. Encode time is proportional to image size and training steps.

Output artifact:

```
capture.cifv2/
├── manifest.json          component digest binding
├── receipt.json           FARD-compatible replay chain
├── preview.png            mandatory thumbnail projection (dumb-codec-visible)
├── canonical_lms.tensor   LMS measurement anchor
├── lambda_real.bin        wavelet coefficients
├── edges.cifedge          structural edge segments
├── procedural.json        texture descriptors
├── inr.siren              trained SIREN weights
└── lambda_mod.bin         modular residue fingerprint
```

### Verify

```bash
cifv2 verify --artifact capture.cifv2
```

Recomputes every digest, checks the manifest, recomputes the modular residue, and confirms the artifact digest against the receipt.

```json
{
  "ok": true,
  "artifact_digest": "sha256:...",
  "steps_verified": 6
}
```

### Render

```bash
cifv2 render --artifact capture.cifv2 --out render.png --width 4096 --height 4096
```

Renders a projection at any target resolution. Composites three layers:

1. Bilinear sample from `canonical_lms.tensor` — base color field
1. Signed-distance edge compositor from `edges.cifedge` — resolution-independent boundaries
1. SIREN residual from `inr.siren` — trained continuous color field

Writes a projection receipt alongside the output:

```
render.png
render.projection.json
```

### Replay

```bash
cifv2 replay --input photo.png --artifact capture.cifv2 --temp-out replay.cifv2
```

Re-encodes the input and confirms the new artifact digest matches the stored receipt. Proves deterministic reproducibility.

-----

## Φ-CIFv2 Operator

```
Φ(I):

1. canonicalize_input(I)        → canonical_lms.tensor
2. wavelet_anchor(T0)           → lambda_real.bin
3. structural_edges(T0)         → edges.cifedge
4. procedural_residual(...)     → procedural.json
5. neural_residual(...)         → inr.siren        ← trained SIREN field
6. modular_residue(lambda_real) → lambda_mod.bin
7. assemble(...)                → manifest.json
8. receipt(...)                 → receipt.json
```

`neural_residual` runs mini-batch SGD against the canonical LMS tensor. The SIREN is fitted to minimize MSE between `f(x,y)` and the target pixel values. Weights are deterministic from a SHA-256 derived seed. The trained parameters are serialized as executable `f32` weights in Base85.

-----

## Render Pipeline

```
for each output pixel (px, py):

  1. map to canonical coordinates (cx, cy)
  2. bilinear sample canonical_lms.tensor        → base LMS
  3. for each edge segment in tile:
       signed distance to cubic segment
       smoothstep influence by contrast and blur_sigma
       accumulate into LMS channels
  4. evaluate inr.siren at (px/width, py/height)  → residual LMS delta
  5. lms_to_srgb(base + edge + siren)             → output pixel
```

Edge segments are spatially accelerated by a 32×32 tile grid. SIREN evaluation is a 4-layer forward pass: 3 sin layers + 1 linear output.

-----

## Artifact Format

### `canonical_lms.tensor`

```
magic:  8 bytes  "CIFLMS1\0"
width:  u32      little-endian
height: u32      little-endian
pixels: f32×3    little-endian [L, M, S] per pixel
```

### `lambda_real.bin` / `lambda_mod.bin`

```
magic:  8 bytes  "CIFI64\0\0"
count:  u64      little-endian
values: i64      little-endian fixed-point, scale 1_000_000
```

`lambda_mod.bin` values are in Z/343Z.

### `edges.cifedge`

JSON array of cubic Bézier edge records. All coordinates fixed-point, scale `1_000_000`:

```json
{
  "x0": 1000000, "y0": 1000000,
  "x1": 5000000, "y1": 2000000,
  "c0x": 3000000, "c0y": 1500000,
  "c1x": 3000000, "c1y": 1500000,
  "width": 1000000,
  "contrast_left": 753429,
  "contrast_right": -753429,
  "blur_sigma": 750000
}
```

### `inr.siren`

```json
{
  "architecture": {
    "input": 2,
    "output": 3,
    "hidden_layers": 3,
    "hidden_width": 32,
    "activation": "sin",
    "omega0": 1.0,
    "steps": 64,
    "learning_rate": 0.001
  },
  "weights_b85": "..."
}
```

Weight layout: `[W0(32×2)+b0(32), W1(32×32)+b1(32), W2(32×32)+b2(32), Wout(3×32)+bout(3)]` — 2307 parameters, little-endian f32, Base85 encoded.

-----

## Determinism

The following are fixed across all platforms:

- sRGB linearization: γ = 2.2
- RGB → LMS matrix (fixed coefficients)
- Canonical resize: Triangle filter, max-side bounded
- Little-endian binary serialization throughout
- Fixed-point scale: `1_000_000`
- Euclidean modular projection: `((x % 343) + 343) % 343`
- SIREN seed: `SHA-256("h_input || h_lambda || h_edges || h_proc || SIREN_INIT")`
- Mini-batch pixel order: deterministic from seed via `DetRng`
- JSON field ordering: `BTreeMap` (lexicographic)
- Digest ordering: stable insertion order in manifest

-----

## Validate

```bash
./scripts/validate.sh
```

Runs build → test → encode → verify → render → replay in sequence. All stages must pass and the artifact digest must be stable across encode and replay.

-----

## Dumb Codec Compatibility

A PNG or JPEG viewer that has no CIF awareness sees exactly one thing:

```
preview.png  →  valid thumbnail raster
```

No metadata. No extensions. No CIF knowledge required. The thumbnail is a standalone PNG produced by the CIF render path at canonical resolution before artifact assembly.

CIF-aware applications open the artifact directory directly and render at any target resolution.
## Validated

Tested against a 256x256 synthetic image with gradients, sharp edges, and texture.

Encode: 1.7s. Verify: 6 steps clean. Replay: identical artifact digest.

Render scale proof:

    artifact_digest  sha256:9fa24044...  invariant across all renders
    render_digest    sha256:c7bbd6f1...  256x256
    render_digest    sha256:cd220afe...  1024x1024
    render_digest    sha256:b2872655...  4096x4096

A 256x256 source rendered to 4096x4096 from the same artifact.
The artifact digest did not change. The render digest is deterministic per resolution.
