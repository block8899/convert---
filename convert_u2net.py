import torch
import pnnx
import os
import sys
import gc
import urllib.request
import subprocess

print("1. Downloading U2NetP (small)...")

MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
MODEL_PATH = "u2netp.onnx"

if not os.path.exists(MODEL_PATH):
    print("   Downloading...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"   Done: {os.path.getsize(MODEL_PATH)/1024/1024:.1f} MB")
else:
    print(f"   Already exists: {os.path.getsize(MODEL_PATH)/1024/1024:.1f} MB")

print("2. Simplifying ONNX...")
SIMPLIFIED = "u2netp_simple.onnx"
subprocess.run([sys.executable, "-m", "onnxsim", MODEL_PATH, SIMPLIFIED], check=True)
print(f"   Simplified: {os.path.getsize(SIMPLIFIED)/1024/1024:.1f} MB")

print("3. Converting ONNX → NCNN...")
dummy = torch.randn(1, 3, 320, 320)
try:
    pnnx.export_onnx(SIMPLIFIED, "u2netp", inputs=dummy)
    print("   PNNX export done!")
except Exception as e:
    print(f"   PNNX failed: {e}")
    sys.exit(1)

del dummy
gc.collect()

print("4. Verifying...")

pf = "u2netp.ncnn.param"
bf = "u2netp.ncnn.bin"

if os.path.exists(pf) and os.path.exists(bf):
    sp = os.path.getsize(pf) / 1024
    sb = os.path.getsize(bf) / 1024
    print(f"  {pf}: {sp:.1f} KB")
    print(f"  {bf}: {sb:.1f} KB")

    with open(pf, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("Input"):
            parts = line.split()
            if len(parts) >= 3:
                print(f"  Input blob: {parts[2]}")
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("7767517"):
            parts = line.split()
            if len(parts) >= 4:
                print(f"  Output blob: {parts[-1]}")
                break

    print(f"\nTotal size: {(sp + sb):.0f} KB")
    print("U2NetP NCNN OK!")
else:
    print("FAILED!")
    print(f"Files: {os.listdir('.')}")
    sys.exit(1)
