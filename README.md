# CIF v2 — Collapse Image Format v2

CIF v2 encodes a raster image into a deterministic, replay-verifiable artifact that renders at any resolution — from thumbnail to billboard — without degradation.

The artifact is the image. Raster files are projections of it.

-----

## How It Works

Standard image formats store pixels. CIF v2 stores the image as an executable field — a set of components that can be evaluated at any target resolution:

- **`canonical_lms.tensor`** — the perceptual measurement anchor. A fixed-size LMS color tensor derived from the source image at canonical resolution. The finite origin of all renders.
- **`edges.cifedge`** — resolution-independent structural boundaries. Cubic Bézier segments in fixed-point coordinates. No raster grid. Renders at full sharpness at any output size.
- **`inr.siren`** — a trained implicit neural field `f(x,y) → (L,M,S)` defined over `[0,1]²`. Continuous. No native resolution. Trained with OKLab perceptual loss. Sampled at the target pixel density at render time.
- **`lambda_real.bin`** — fixed-point wavelet coefficients derived from the LMS tensor.
- **`procedural.json`** — deterministic one-over-f texture descriptors.
- **`manifest.json`** — SHA-256 digest binding for all components.
- **`receipt.json`** — FARD-compatible replay-verifiable step chain.
- **`preview.png`** — mandatory thumbnail projection. The only component visible to dumb PNG/JPEG decoders.

-----

## Projection Contract

Every artifact has a stable identity:

```
artifact_digest   SHA-256 of the source artifact — invariant across all renders
render_digest     SHA-256 of a rendered PNG — deterministic given (artifact, width, height)
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

The artifact digest never changes. Pixels are outputs, not identity.

-----

## Dumb Codec Compatibility

A PNG or JPEG viewer with no CIF awareness sees exactly one thing:

```
preview.png  →  valid thumbnail raster
```

No metadata. No extensions. No CIF knowledge required. CIF-aware applications open the artifact directory or `.cifv2f` file directly and render at any resolution.

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
cifv2 encode --input photo.jpg --out capture.cifv2
cifv2 encode --input photo.png --out capture.cifv2 --max-side 512
```

Runs the full Φ operator: canonicalize → wavelet → edges → procedural → SIREN training → assemble → receipt. Output is an artifact directory.

### Verify

```bash
cifv2 verify --artifact capture.cifv2
cifv2 verify --artifact capture.cifv2f
```

Recomputes every component digest, checks the manifest, recomputes the modular residue from `lambda_real.bin` in memory, and confirms the artifact digest against the receipt.

### Render

```bash
cifv2 render --artifact capture.cifv2 --out render.png --width 4096 --height 4096
```

Renders a projection at any target resolution. Accepts both directory artifacts and `.cifv2f` single files. Composites three layers:

1. Bilinear sample from `canonical_lms.tensor` — base color field
1. Signed-distance edge compositor from `edges.cifedge` — resolution-independent boundaries
1. SIREN residual from `inr.siren` — trained continuous color field (OKLab loss)

Writes a projection receipt alongside the output PNG.

### Replay

```bash
cifv2 replay --input photo.jpg --artifact capture.cifv2 --temp-out replay.cifv2
```

Re-encodes the input from scratch and confirms the new artifact digest matches the stored receipt. Proves deterministic reproducibility end-to-end.

### Pack / Unpack

```bash
cifv2 pack   --artifact capture.cifv2  --out capture.cifv2f
cifv2 unpack --file capture.cifv2f     --out capture.cifv2
```

Packs an artifact directory into a single distributable `.cifv2f` container file. `verify`, `render`, and `replay` accept both forms transparently — the artifact digest is invariant between them.

-----

## Artifact Size

All binary components are zstd-compressed. Edge records use a binary format instead of JSON. Measured against a real 5472×3648 camera JPEG:

```
canonical_lms.tensor    513KB → 44KB    (zstd, 11.6×)
edges.cifedge           438KB → 29KB    (binary + zstd, 15×)
lambda_real.bin         1.3MB → 169KB   (zstd, 7.7×)
lambda_mod.bin          1.3MB → 0       (removed — recomputed at verify time)
inr.siren                11KB           (Base85 f32 weights)
preview.png              64KB           (PNG thumbnail)

directory artifact       336KB
.cifv2f container        320KB
```

-----

## Single-File Container Format

The `.cifv2f` container is a binary format with an indexed header:

```
[0..8]    magic       "CIFV2\x00\x00\x00"
[8..12]   version     u32 little-endian = 1
[12..16]  n_entries   u32 little-endian
[16..]    index       n_entries × 64-byte descriptors
[...]     data        concatenated compressed components
```

Each index entry: 32-byte null-padded name, u64 offset, u64 length, 16 bytes reserved.

-----

## Φ-CIFv2 Operator

```
Φ(I):

1. canonicalize_input(I)        → canonical_lms.tensor
2. wavelet_anchor(T0)           → lambda_real.bin
3. structural_edges(T0)         → edges.cifedge
4. procedural_residual(...)     → procedural.json
5. neural_residual(...)         → inr.siren        (OKLab perceptual loss)
6. modular_residue(lambda_real) → digest only, not stored
7. assemble(...)                → manifest.json
8. receipt(...)                 → receipt.json
```

-----

## Render Pipeline

```
for each output pixel (px, py):
  1. map to canonical coordinates
  2. bilinear sample canonical_lms.tensor       → base LMS
  3. for each edge in tile (32×32 tile grid):
       signed distance to cubic segment
       smoothstep by contrast + blur_sigma
       accumulate into LMS channels
  4. evaluate inr.siren at (px/w, py/h)         → residual LMS delta
  5. lms_to_srgb(base + edges + siren)          → output pixel
```

-----

## SIREN Architecture

```json
{
  "input": 2,
  "output": 3,
  "hidden_layers": 3,
  "hidden_width": 32,
  "activation": "sin",
  "omega0": 1.0,
  "steps": 64,
  "learning_rate": 0.001
}
```

Weight layout: `[W0(32×2)+b0(32), W1(32×32)+b1(32), W2(32×32)+b2(32), Wout(3×32)+bout(3)]` — 2307 parameters, little-endian f32, Base85 encoded. Trained with mini-batch SGD, loss computed in OKLab perceptual color space.

-----

## Determinism

The following are fixed across all platforms and runs:

- sRGB linearization: γ = 2.2
- RGB → LMS matrix (fixed Hunt-Pointer-Estévez coefficients)
- Canonical resize: Triangle filter, max-side bounded
- Little-endian binary serialization throughout
- Fixed-point coordinate scale: `1_000_000`
- Euclidean modular projection: `((x % 343) + 343) % 343`
- SIREN seed: `SHA-256(h_input || h_lambda || h_edges || h_proc || "SIREN_INIT")`
- Mini-batch pixel sampling: deterministic from seed via `DetRng`
- JSON field ordering: `BTreeMap` (lexicographic)
- Digest ordering: stable insertion order in manifest

-----

## Proven Results

Tested against a real 5472×3648 camera JPEG (Wolverine scanner, Exif metadata):

```
encode           1.3s
verify           ok, 6 steps
replay           identical artifact digest
render 256×256   render_digest stable and reproducible
render 1024×1024 artifact_digest invariant
render 4096×4096 artifact_digest invariant  ← billboard scale
```

The artifact digest is invariant across all render resolutions. Each render digest is deterministic given `(artifact, width, height)`.

-----

## Validate

```bash
./scripts/validate.sh
```

Runs: build → test → encode → verify → render (256 / 1024 / 4096) → replay → invariance check. All stages must pass. The artifact digest must be stable across encode and replay.
## Browser Viewer

CIF v2 includes a browser-based viewer that runs entirely in WebAssembly — no server, no install.

    open viewer.html in any modern browser
    drop a .cifv2f file onto the viewer
    drag the zoom slider from 1x to 16x

The artifact digest is displayed in the header and remains identical at every zoom level.
This is the core property: the image identity does not change with render resolution.

The WASM renderer (134KB) implements the full read path:

    unpack .cifv2f container
    decompress canonical_lms.tensor (ruzstd, pure Rust)
    bilinear sample at target resolution
    lms_to_srgb conversion

Build the WASM package:

    cd wasm
    wasm-pack build --target web --release

Serve locally:

    python3 -m http.server 8081
    open http://localhost:8081/viewer.html

## CIF-RDO v0

CIF-RDO is the optimizer layer above CIF v2. Where CIF v2 proves executable image identity, CIF-RDO proves deterministic representation selection.

See docs/CIF_RDO_v0.md for the frozen specification.

### Objective

For each 32x32 tile, CIF-RDO selects the encoder minimizing:

    J(E,R) = rate_bits(E,R) + quality_lambda * D_oklab(E,R)

J is computed in fixed-point integer arithmetic. Selection is deterministic and reproducible.

### Encoders (v0)

    constant_lms     96 bits   mean LMS value
    affine_lms      576 bits   linear fit over (x, y)
    quadratic_lms  1152 bits   quadratic fit over (1, x, y, x^2, y^2, xy)

### CLI

    cifv2 rdo-encode --input photo.jpg --out photo.cifrdo --tile 32 --quality 1.0

### Encoder Distribution

At quality=1.0 on a 256x256 natural scene (8x8 tile grid):

    constant_lms    34 tiles  (53.1%)  flat regions
    affine_lms      30 tiles  (46.9%)  gradient regions
    quadratic_lms    0 tiles  (0.0%)   rate too high at this quality

At quality=10.0:

    affine_lms      57 tiles  (89.1%)
    constant_lms     5 tiles  ( 7.8%)
    quadratic_lms    2 tiles  ( 3.1%)  earns its bits on complex tiles

The solver activates higher-cost encoders only when distortion savings justify the rate increase. This is the correct behavior.

### Artifact Layout

    photo.cifrdo/
      manifest.json
      receipt.json
      regions/
        tree.json      tile grid, encoder selections, J scores
        payloads.bin   concatenated tile payloads

### Core Invariant

    same input + same quality_lambda = same artifact_digest

## CIF-RDO v0 — Current Status

rdo-encode, rdo-verify, and rdo-render are all working end-to-end.

### Validated results

Encoded bench/inputs/natural_scene.png (256x256) at quality=1.0:

    artifact_digest  sha256:cdefd8e6...  invariant across all renders
    render_digest    sha256:a3db5d5f...  256x256
    render_digest    sha256:c9f3f719...  1024x1024

rdo-verify confirms 64/64 region payload digests and the full digest chain.

### CLI summary

    cifv2 rdo-encode --input photo.jpg --out photo.cifrdo --tile 32 --quality 1.0
    cifv2 rdo-verify --artifact photo.cifrdo
    cifv2 rdo-render --artifact photo.cifrdo --out render.png --width 1024 --height 1024

### What the solver is doing

At quality=1.0, rate cost dominates. constant_lms wins on flat regions.
At quality=10.0, distortion cost dominates. affine_lms wins on gradients.
quadratic_lms activates only when it reduces distortion enough to justify 1152 bits over 576.

This is the correct behavior of a J-minimizer under the stated objective.

## CIF-RDO v0 — Encoder Results

Four encoders are implemented and content-adaptive selection is proven.

### Encoder behavior by content type

Smooth gradients (quality=1.0):

    constant_lms    34 tiles  (53.1%)
    affine_lms      30 tiles  (46.9%)
    wavelet_tile     0 tiles  (0.0%)

High-frequency checkerboard (quality=1.0):

    wavelet_tile    64 tiles  (100.0%)

The solver selects the right encoder for each tile without any heuristics.
constant wins on flat regions because its 96-bit rate cost is unbeatable.
affine wins on gradients because it fits them at 576 bits with low distortion.
wavelet wins on high-frequency content because detail subbands capture it efficiently.
quadratic activates only at high quality lambda where distortion cost dominates rate cost.

### Wavelet encoder

Single-level 2D Haar transform per channel on each 32x32 tile.
Coefficients quantized and zero-RLE encoded. Variable payload length.
rate_bits = payload length * 8.
