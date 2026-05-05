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
    quadratic_lms     576 bits    quadratic fit over (1, x, y, x^2, y^2, xy)  3ch x 6 f32
    wavelet_tile     variable     Haar + quality-adaptive quantization + RLE + zstd
    edge_tile        variable     background mean + Sobel edge segments
    micro_siren_tile 1632 bits    micro SIREN (1 hidden layer, 8 wide) per tile
    dct_tile          840 bits    32x32 DCT-II, K=16 zigzag coefficients per channel

Wavelet and DCT quantization scales with quality: higher quality_lambda = finer quantization = lower distortion = larger payload.

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

    image              tiles  const affine  quad  wave  edge siren   dct  size_kb   ms
    diagonal_edge         64     56      0     0     8     0     0     0        1  151
    high_frequency        64      0      0     0    64     0     0     0        1  124
    low_frequency         64     16     29    19     0     0     0     0        3  123
    natural_scene         64      4     24     0     1     0     0    35        5  123
    sharp_synthetic       64     64      0     0     0     0     0     0        1  132
    smooth_gradient       64      0     64     0     0     0     0     0        2  123
    vertical_edge         64     56      0     0     0     0     0     8        1  121
    TOTAL                448    196    117    19    73     0     0    43       16  897

DCT displaced quadratic on natural_scene (35 dct vs 34 quad previously). Size dropped from 11KB to 5KB at equal distortion.

### External codec comparison (quality=10.0 vs avif/jxl/webp at quality=60)

    image           codec    size_kb   D_oklab
    diagonal_edge   cif-rdo      1     0.000001   beats avif (D=0.000006) at equal size
                    avif          1     0.000006
                    jxl           1     0.000053

    sharp_synthetic cif-rdo      1     0.000000   zero distortion
                    avif          0     0.000002

    smooth_gradient cif-rdo      1     0.000000   zero distortion
                    avif          1     0.000005

    natural_scene   cif-rdo      5     0.000589   2.5x larger than avif, 27x worse D
                    avif          2     0.000022
                    jxl           2     0.000039

    high_frequency  cif-rdo      1     0.000150   competitive with jxl
                    avif          1     0.000000
                    jxl           2     0.000052

### Assessment

CIF-RDO beats or matches AVIF at quality=10 on structured and synthetic content. The remaining gap is on natural images with complex local color variation — no current encoder represents smooth nonlinear color gradients compactly enough to match AVIF's global entropy coding.

What works:
- Deterministic region selection under a fixed objective
- Content-adaptive encoder choice without heuristics
- Artifact digest invariant across render resolutions
- Zero D_oklab on flat and linear content
- Beats AVIF on diagonal_edge, sharp_synthetic, smooth_gradient, vertical_edge
- DCT encoder handles natural image gradients better than quadratic

What does not yet work:
- Distortion parity with AVIF on natural images at equal size
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
          wavelet.rs       quality-adaptive quantization
          edge.rs
          siren.rs
          dct.rs           32x32 DCT-II, K=16 zigzag
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
