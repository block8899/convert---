import os
import sys
import subprocess
import urllib.request

print("1. Downloading U2NetP...")

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

print("3. Converting ONNX → NCNN via pnnx CLI...")
# ★ Dùng pnnx command line, không phải Python API
result = subprocess.run(
    ["pnnx", SIMPLIFIED, "inputshape=1,3,320,320"],
    capture_output=True, text=True, timeout=300
)
print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

# pnnx CLI tạo file theo tên input: u2netp_simple.ncnn.param
pf1 = "u2netp_simple.ncnn.param"
bf1 = "u2netp_simple.ncnn.bin"
pf2 = "u2netp.ncnn.param"
bf2 = "u2netp.ncnn.bin"

# Rename nếu cần
if os.path.exists(pf1) and os.path.exists(bf1):
    os.rename(pf1, pf2)
    os.rename(bf1, bf2)

print("4. Verifying...")

pf = pf2
bf = bf2

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

    print(f"\nTotal: {(sp + sb):.0f} KB")
    print("U2NetP NCNN OK!")
else:
    # Check what pnnx created
    ncnn_files = [f for f in os.listdir('.') if 'ncnn' in f]
    print(f"FAILED! ncnn files found: {ncnn_files}")
    all_files = os.listdir('.')
    print(f"All files: {all_files}")
    sys.exit(1)
