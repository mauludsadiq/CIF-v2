# CIF v2 — Collapse Image Format

CIF v2 encodes a raster image into a deterministic, replay-verifiable artifact that renders at any resolution without degradation. CIF-RDO is the optimizer layer above it: a deterministic rate-distortion minimizer that selects the best analytical representation for every tile of an image under a fixed objective function.

The artifact is the image. Raster files are projections of it.

---

## Architecture

CIF v2 and CIF-RDO are two distinct layers.

**CIF v2** proves executable image identity. Given an image, it produces a verifiable artifact with a stable digest that is invariant across all render resolutions. The artifact encodes the image as a set of continuous fields — a perceptual tensor, edge segments, wavelet coefficients, and a trained implicit neural field.

**CIF-RDO** proves deterministic representation selection. Given an image, it partitions it into 32x32 tiles and selects the encoder that minimizes a fixed objective function for each tile. The selection is reproducible: the same input at the same quality lambda always produces the same artifact digest.

---

## CIF v2

### Artifact components

    canonical_lms.tensor   LMS color tensor, zstd-compressed f32
    edges.cifedge          cubic Bezier edge segments, binary + zstd
    inr.siren              trained SIREN weights (OKLab perceptual loss)
    lambda_real.bin        fixed-point Haar wavelet coefficients, zstd
    procedural.json        deterministic one-over-f texture descriptors
    manifest.json          SHA-256 digest binding for all components
    receipt.json           FARD-compatible replay-verifiable step chain
    preview.png            thumbnail projection

### Projection contract

    artifact_digest   invariant across all renders
    render_digest     deterministic given (artifact, width, height)

### CLI

    cifv2 encode  --input photo.jpg --out capture.cifv2
    cifv2 verify  --artifact capture.cifv2
    cifv2 render  --artifact capture.cifv2 --out render.png --width 4096 --height 4096
    cifv2 replay  --input photo.jpg --artifact capture.cifv2 --temp-out replay.cifv2
    cifv2 pack    --artifact capture.cifv2 --out capture.cifv2f
    cifv2 unpack  --file capture.cifv2f --out capture.cifv2

### Browser viewer

Drop a .cifv2f file onto viewer.html. No server, no install — pure WASM. The artifact digest is displayed in the header and remains identical at every zoom level from 1x to 16x.

    cd wasm && wasm-pack build --target web --release
    python3 -m http.server 8081
    open http://localhost:8081/viewer.html

### Proven results

Tested against a 5472x3648 camera JPEG:

    source JPEG          3.7MB
    artifact             336KB    (-91%)
    encode time          1.3s
    artifact_digest      invariant at 256x256, 1024x1024, 4096x4096
    replay               identical artifact digest from scratch

---

## CIF-RDO

### Objective

For each 32x32 LMS tile, CIF-RDO selects the encoder minimizing:

    J(E,R) = rate_bits(E,R) + quality_lambda x D_oklab(E,R)

J is computed in fixed-point integer arithmetic. Ties broken by rate, then encoder name, then payload digest. Selection is fully deterministic.

See docs/CIF_RDO_v0.md for the frozen specification.

### Encoders

    constant_lms       96 bits    mean LMS value
    affine_lms        576 bits    linear fit over (x, y)
    quadratic_lms     576 bits    quadratic fit over (1, x, y, x^2, y^2, xy)
    wavelet_tile     variable     Haar transform + quantization + RLE
    edge_tile        variable     background mean + Sobel edge segments
    micro_siren_tile 1632 bits    micro SIREN (1 hidden layer, 8 wide) per tile

### CLI

    cifv2 rdo-encode --input photo.jpg --out photo.cifrdo --tile 32 --quality 1.0
    cifv2 rdo-verify --artifact photo.cifrdo
    cifv2 rdo-render --artifact photo.cifrdo --out render.png --width 1024 --height 1024
    cifv2 rdo-bench  --input bench/inputs --quality 1.0
    cifv2 rdo-bench  --input bench/inputs --quality 1.0 --compare

### Artifact layout

    photo.cifrdo/
      manifest.json
      receipt.json
      regions/
        tree.bin           26-byte header + 5 bytes per tile
        payloads.bin.zst   zstd-compressed tile payloads

tree.bin stores the tile grid compactly: 346 bytes for 64 tiles vs 30KB for the equivalent tree.json (87x reduction). payloads.bin.zst applies zstd level 3 to the concatenated tile payloads.

### Benchmark results (quality=1.0, 7 images, 448 tiles)

    image              tiles  const affine  quad  wave  edge siren  size_kb   ms
    diagonal_edge         64     56      0     8     0     0     0        1   25
    high_frequency        64      0      0     0    64     0     0        2   27
    low_frequency         64     64      0     0     0     0     0        2   25
    natural_scene         64     30      0    34     0     0     0        4   27
    sharp_synthetic       64     64      0     0     0     0     0        1   24
    smooth_gradient       64     64      0     0     0     0     0        2   26
    vertical_edge         64     56      0     8     0     0     0        1   25
    TOTAL                448    334      0    50    64     0     0       13  179

Encoder behavior across quality levels:

    quality=1.0    const 334  affine   0  quad  50  wave  64
    quality=10.0   const 196  affine 117  quad  44  wave  91
    quality=100.0  const 180  affine 117  quad  35  wave 116

### External codec comparison (quality=1.0)

    image           codec    size_kb   D_oklab
    natural_scene   cif-rdo      3     0.001501
                    avif          2     0.000022
                    jxl           2     0.000039
                    webp          1     0.000086

    sharp_synthetic cif-rdo      1     0.000000   zero distortion, equal size to avif
                    avif          0     0.000002

    high_frequency  cif-rdo      2     0.011955   was 121KB before zstd payloads
                    avif          1     0.000000
                    jxl           2     0.000052

    diagonal_edge   cif-rdo      1     0.001700
                    avif          1     0.000006

Lossless comparison (natural_scene 256x256):

    source PNG         7.2KB
    CIF-RDO q=1        3KB    (lossy, D_oklab=0.001501)
    JXL lossless       7.8KB
    AVIF lossless       17KB

### Assessment

CIF-RDO is size-competitive with AVIF and JXL across all content types after zstd payload compression. The remaining gap is distortion, not size. On natural images CIF-RDO produces higher D_oklab than AVIF at equal file size. On flat and synthetic content it achieves zero distortion at competitive size.

What works:
- Deterministic region selection under a fixed objective
- Content-adaptive encoder choice without heuristics
- Artifact digest invariant across render resolutions
- Zero D_oklab on flat and linear content
- Size-competitive with AVIF and JXL on all tested content

What does not yet work:
- Distortion parity with AVIF on natural images
- edge_tile and micro_siren_tile not yet activating on corpus

---

## Build

    cargo build --release

Binary: ./target/release/cifv2

External codec comparison requires avifenc, avifdec, cjxl, djxl, cwebp, dwebp.

---

## Repository structure

    src/
      main.rs              CIF v2 CLI and core pipeline
      rdo/
        types.rs           LmsTile, EncodedRegion, RegionEncoder trait
        objective.rs       D_oklab fixed-point, J_fixed
        encoders/
          constant.rs
          affine.rs
          quadratic.rs
          wavelet.rs
          edge.rs
          siren.rs
        select.rs          argmin loop with 4-rule tie-breaking
        encode.rs          rdo-encode
        verify.rs          rdo-verify
        render.rs          rdo-render
        bench.rs           rdo-bench
        compare.rs         external codec comparison
        tree_bin.rs        compact binary tree format
    docs/
      CIF_RDO_v0.md        frozen v0 specification
    wasm/
      src/lib.rs           WASM viewer
      viewer.html          browser entry point
    bench/
      inputs/              corpus images
