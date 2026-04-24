#!/usr/bin/env bash
set -euo pipefail

cargo build --release
cargo test --release

python3 - << 'PY'
from PIL import Image
Image.new("RGB", (64,64), (255,0,0)).save("test.png")
PY

./target/release/cifv2 encode --input test.png --out capture.cifv2
./target/release/cifv2 verify --artifact capture.cifv2
./target/release/cifv2 render --artifact capture.cifv2 --out render.png --width 256 --height 256
./target/release/cifv2 replay --input test.png --artifact capture.cifv2 --temp-out replay_tmp.cifv2
