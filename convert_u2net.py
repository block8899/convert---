import torch
import torch.nn as nn
import pnnx
import os
import sys
import gc
import urllib.request

print("1. Downloading U2Net model...")

MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
MODEL_PATH = "u2net.onnx"

if not os.path.exists(MODEL_PATH):
    print(f"   Downloading...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"   Done: {os.path.getsize(MODEL_PATH)/1024/1024:.1f} MB")
else:
    print(f"   Already exists: {os.path.getsize(MODEL_PATH)/1024/1024:.1f} MB")

print("2. Converting ONNX → NCNN...")

import onnx
from onnx import simplifier

print("   Loading ONNX...")
model = onnx.load(MODEL_PATH)
print(f"   Nodes: {len(model.graph.node)}")

print("   Simplifying...")
model_simp = onnx.simplify.simplify(model)
if isinstance(model_simp, tuple):
    model_simp = model_simp[0]

SIMPLIFIED = "u2net_simple.onnx"
onnx.save(model_simp, SIMPLIFIED)
print(f"   Saved simplified: {os.path.getsize(SIMPLIFIED)/1024/1024:.1f} MB")

print("   Converting with PNNX...")
dummy = torch.randn(1, 3, 320, 320)
try:
    pnnx.export_onnx(SIMPLIFIED, "u2net", inputs=dummy)
    print("   PNNX export done!")
except Exception as e:
    print(f"   PNNX failed: {e}")
    sys.exit(1)

del dummy
gc.collect()

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

    print(f"\nTotal size: {sp + sb*1024:.0f} KB")
    print("U2Net NCNN OK!")
else:
    print("FAILED!")
    print(f"Files: {os.listdir('.')}")
    sys.exit(1)
