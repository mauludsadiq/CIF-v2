#!/usr/bin/env bash
set -euo pipefail

cargo build --release
cargo test --release

# --- Test 1: solid red 64x64 (baseline, edges=[]) ---
python3 - << 'PY'
from PIL import Image
Image.new("RGB", (64,64), (255,0,0)).save("test.png")
PY

./target/release/cifv2 encode --input test.png --out capture.cifv2
./target/release/cifv2 verify --artifact capture.cifv2
./target/release/cifv2 render --artifact capture.cifv2 --out render.png --width 256 --height 256
./target/release/cifv2 replay --input test.png --artifact capture.cifv2 --temp-out replay_tmp.cifv2

# --- Test 2: structured image with edges and trained SIREN ---
python3 - << 'PY'
from PIL import Image, ImageDraw, ImageFilter
import math
img = Image.new("RGB", (256, 256))
px = img.load()
for y in range(256):
    for x in range(256):
        r = int(30 + (x/256)*80 + (y/256)*40)
        g = int(80 + (x/256)*60 + (y/256)*30)
        b = int(180 - (y/256)*60)
        px[x,y] = (min(r,255), min(g,255), min(b,255))
draw = ImageDraw.Draw(img)
draw.rectangle([20, 180, 120, 250], fill=(200, 160, 80))
draw.rectangle([130, 190, 230, 250], fill=(160, 120, 60))
draw.polygon([(20,180),(70,110),(120,180)], fill=(140,60,40))
draw.polygon([(130,190),(180,115),(230,190)], fill=(120,50,35))
draw.rectangle([40,195,75,230], fill=(180,210,240))
draw.rectangle([148,205,175,235], fill=(180,210,240))
draw.ellipse([190,20,240,70], fill=(255,240,100))
for y in range(245, 256):
    for x in range(256):
        v = int(60 + 20*math.sin(x*0.3) + 15*math.sin(x*0.7+y))
        px[x,y] = (v+30, v+40, v)
img = img.filter(ImageFilter.SMOOTH)
img.save("test_photo.png")
PY

./target/release/cifv2 encode --input test_photo.png --out capture_photo.cifv2
./target/release/cifv2 verify --artifact capture_photo.cifv2
./target/release/cifv2 render --artifact capture_photo.cifv2 --out render_photo_256.png --width 256 --height 256
./target/release/cifv2 render --artifact capture_photo.cifv2 --out render_photo_1024.png --width 1024 --height 1024
./target/release/cifv2 render --artifact capture_photo.cifv2 --out render_photo_4096.png --width 4096 --height 4096
./target/release/cifv2 replay --input test_photo.png --artifact capture_photo.cifv2 --temp-out replay_photo.cifv2

# --- Invariance check ---
python3 - << 'PY'
import json, sys
a = json.load(open("render_photo_256.projection.json"))
b = json.load(open("render_photo_1024.projection.json"))
c = json.load(open("render_photo_4096.projection.json"))
ad = a["artifact_digest"]
if not (ad == b["artifact_digest"] == c["artifact_digest"]):
    print("FAIL: artifact_digest not invariant across resolutions")
    sys.exit(1)
if len({a["projection"]["render_digest"], b["projection"]["render_digest"], c["projection"]["render_digest"]}) != 3:
    print("FAIL: render_digests not distinct across resolutions")
    sys.exit(1)
print("artifact_digest invariant across 256 / 1024 / 4096: OK")
print("render_digest distinct per resolution: OK")
PY
