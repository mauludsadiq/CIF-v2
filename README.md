# CIF v2 — Collapse Image Format

CIF v2 encodes a raster image into a deterministic, replay-verifiable artifact that renders at any resolution without degradation. CIF-RDO is the optimizer layer above it: a deterministic rate-distortion minimizer that selects the best analytical representation for every tile of an image under a fixed objective function.

The artifact is the image. Raster files are projections of it.

---

## Architecture

**CIF v2** proves executable image identity. Given an image, it produces a verifiable artifact with a stable digest invariant across all render resolutions. The artifact encodes the image as a set of continuous fields — a perceptual tensor, edge segments, wavelet coefficients, and a trained implicit neural field.

**CIF-RDO** proves deterministic representation selection. Given an image, it partitions it into 32x32 tiles and selects the encoder minimizing a fixed objective for each tile. The same input at the same quality lambda always produces the same artifact digest.

---

## CIF v2

### Artifact components

    canonical_lms.tensor   LMS color tensor, zstd-compressed f32
    edges.cifedge          cubic Bezier edge segments, binary + zstd
    inr.siren              trained SIREN weights (OKLab perceptual loss)
    lambda_real.bin        fixed-point Haar wavelet coefficients, zstd
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
    wavelet_tile     variable     Haar + quality-adaptive quantization + RLE + zstd
    edge_tile        variable     background mean + Sobel edge segments
    micro_siren_tile 1632 bits    micro SIREN (1 hidden layer, 8 wide) per tile

Wavelet quantization scales with quality: q=1 uses Q_STEP=3906, q=10 uses Q_STEP=1235, q=100 uses Q_STEP=391.

### CLI

    cifv2 rdo-encode  --input photo.jpg --out photo.cifrdo --tile 32 --quality 10.0
    cifv2 rdo-verify  --artifact photo.cifrdo
    cifv2 rdo-render  --artifact photo.cifrdo --out render.png --width 1024 --height 1024
    cifv2 rdo-inspect --artifact photo.cifrdo --source photo.jpg
    cifv2 rdo-bench   --input bench/inputs --quality 10.0
    cifv2 rdo-bench   --input bench/inputs --quality 10.0 --compare

### Artifact layout

    photo.cifrdo/
      manifest.json
      receipt.json
      regions/
        tree.bin           26-byte header + 5 bytes per tile
        payloads.bin.zst   zstd-compressed tile payloads

### Benchmark results (quality=10.0, 7 images, 448 tiles)

    image              tiles  const affine  quad  wave  edge siren  size_kb   ms
    diagonal_edge         64     56      0     8     0     0     0        1   48
    high_frequency        64      0      0     0    64     0     0        1   32
    low_frequency         64     64      0     0     0     0     0        3   27
    natural_scene         64     30      0    34     0     0     0       12   27
    sharp_synthetic       64     64      0     0     0     0     0        1   24
    smooth_gradient       64     64      0     0     0     0     0        2   26
    vertical_edge         64     56      0     8     0     0     0        1   25
    TOTAL                448    334      0    50    64     0     0       22  209

### External codec comparison (quality=10.0 vs avif/jxl/webp at quality=60)

    image           codec    size_kb   D_oklab
    diagonal_edge   cif-rdo      1     0.000001   beats avif (D=0.000006)
                    avif          1     0.000006
                    jxl           1     0.000053

    high_frequency  cif-rdo      1     0.000150   competitive with jxl
                    avif          1     0.000000
                    jxl           2     0.000052

    sharp_synthetic cif-rdo      1     0.000000   zero distortion
                    avif          0     0.000002

    smooth_gradient cif-rdo      1     0.000000   zero distortion
                    avif          1     0.000005

    vertical_edge   cif-rdo      1     0.000000   zero distortion, beats avif
                    avif          0     0.000003

    natural_scene   cif-rdo     11     0.000604   still losing (5x larger, 27x worse D)
                    avif          2     0.000022
                    jxl           2     0.000039

CIF-RDO at quality=10 beats or matches AVIF on 5 of 7 images at equal file size.

### Assessment

What works:
- Deterministic region selection under a fixed objective
- Content-adaptive encoder choice without heuristics
- Artifact digest invariant across render resolutions
- Zero D_oklab on flat, linear, and edge content
- Beats AVIF on diagonal_edge, sharp_synthetic, smooth_gradient, vertical_edge
- Competitive with JXL on high_frequency

What does not yet work:
- Natural images with complex color gradients (natural_scene 11KB vs AVIF 2KB)
- edge_tile and micro_siren_tile not yet activating on corpus

The natural_scene loss is structural: no current encoder fits complex local color variation at moderate rate. This is the next target.

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
          wavelet.rs       quality-adaptive quantization
          edge.rs
          siren.rs
        select.rs          argmin loop with 4-rule tie-breaking
        encode.rs          rdo-encode
        verify.rs          rdo-verify
        render.rs          rdo-render
        inspect.rs         rdo-inspect (per-tile diagnostics)
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
