import torch
import pnnx
import os
import sys
import gc
import urllib.request

print("1. Downloading U2Net model...")

# U2Net standard (small version ~4MB)
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
MODEL_PATH = "u2net.onnx"

if not os.path.exists(MODEL_PATH):
    print(f"   Downloading from {MODEL_URL}...")
    urllib.request.urlretrieve(MODEL_PATH)
    print("   Download done!")
else:
    print("   Already downloaded")

print("2. Converting ONNX to NCNN...")

# pnnx can convert ONNX directly
try:
    import onnx
    print("   Loading ONNX model...")
    model = onnx.load(MODEL_PATH)
    print(f"   ONNX model loaded: {len(model.graph.node)} nodes")

    # Convert via pnnx
    pnnx.export_onnx(MODEL_PATH, "u2net")
    print("   PNNX export done!")
except Exception as e:
    print(f"   ONNX convert failed: {e}")

    # Fallback: try direct conversion
    print("   Trying alternative method...")
    try:
        os.system(f"pnnx {MODEL_PATH}")
        print("   Direct convert done!")
    except Exception as e2:
        print(f"   All methods failed: {e2}")
        sys.exit(1)

print("3. Verifying...")

pf = "u2net.ncnn.param"
bf = "u2net.ncnn.bin"

if os.path.exists(pf) and os.path.exists(bf):
    sp = os.path.getsize(pf) / 1024
    sb = os.path.getsize(bf) / 1024 / 1024
    print(f"  {pf}: {sp:.1f} KB")
    print(f"  {bf}: {sb:.1f} MB")

    with open(pf, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("Input"):
            print(f"  Input: {line}")
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("7767517"):
            parts = line.split()
            if len(parts) >= 4:
                print(f"  Output: {line}")
                break

    print("U2Net NCNN OK!")
else:
    print("FAILED!")
    print(f"Files: {os.listdir('.')}")
    sys.exit(1)
