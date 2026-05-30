"""
ToonClip ONNX → NCNN (via PNNX) - Fixed inputshape mismatch
✅ Now uses [1,3,512,512] to balance compatibility & model size
"""
import os, sys, subprocess, shutil, onnx

def log(msg, level="INFO"):
    print(f"[{level}] {msg}")

def main():
    log("=== ToonClip ONNX → NCNN (via PNNX) ===")

    onnx_file = "ToonClip.onnx"
    if not os.path.exists(onnx_file):
        log(f"MISSING: {onnx_file}", "ERROR"); sys.exit(1)

    log(f"Input: {onnx_file} ({os.path.getsize(onnx_file)/1024/1024:.2f} MB)")

    # 1. Inspect
    log("1. Inspecting ONNX...")
    model = onnx.load(onnx_file)
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    in_shape = [d.dim_value if d.dim_value else -1 for d in model.graph.input[0].type.tensor_type.shape.dim]
    log(f"   ONNX Input: '{in_name}' shape={in_shape}")  # Will show [-1,3,1024,1024]

    # 2. Simplify
    log("2. Simplifying ONNX...")
    sim_file = "toonclip_sim.onnx"
    try:
        subprocess.run([sys.executable, "-m", "onnxsim", onnx_file, sim_file], check=True, capture_output=True)
        log(f"   Simplified: {os.path.getsize(sim_file)/1024/1024:.2f} MB")
    except:
        log("onnxsim failed, using original", "WARN")
        shutil.copy(onnx_file, sim_file)

    # 3. Convert via PNNX - ✅ FIXED INPUTSHAPE
    log("3. Converting via PNNX...")
    
    # ✅ OPTION A: Dùng shape gốc (an toàn nhất)
    # input_shape = "[1,3,1024,1024]"
    
    # ✅ OPTION B: Dùng 512x512 (nhẹ hơn, vẫn tương thích UNet++)
    input_shape = "[1,3,512,512]"
    
    pnnx_cmd = [
        "pnnx", sim_file,
        f"inputshape={input_shape}",
        "device=cpu",
        "fp16=0",
        "optlevel=2"
    ]

    try:
        res = subprocess.run(pnnx_cmd, capture_output=True, text=True, timeout=300)
        if res.stdout: log(f"   PNNX stdout (tail):\n{res.stdout[-400:]}")
        if res.returncode != 0:
            log(f"PNNX FAILED!\n{res.stderr[-800:]}", "ERROR"); sys.exit(1)
    except subprocess.TimeoutExpired:
        log("PNNX timed out", "ERROR"); sys.exit(1)

    # 4. Verify & extract blob names
    pnnx_param = "toonclip_sim.ncnn.param"
    pnnx_bin = "toonclip_sim.ncnn.bin"
    if not os.path.exists(pnnx_param) or not os.path.exists(pnnx_bin):
        log("PNNX output missing!", "ERROR"); sys.exit(1)

    log(f"   Converted: param={os.path.getsize(pnnx_param)/1024:.1f} KB, bin={os.path.getsize(pnnx_bin)/1024/1024:.2f} MB")

    # Extract blob names
    log("4. Detecting NCNN blob names...")
    input_blob = output_blob = None
    with open(pnnx_param, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    for line in lines:
        if line.startswith("Input"): input_blob = line.split()[-1]
        output_blob = line.split()[-1]  # Last layer output
    if input_blob and output_blob:
        log(f"   ✅ NCNN Input: '{input_blob}', Output: '{output_blob}'")
        log(f"   👉 Update C++: ex.input(\"{input_blob}\", in) / ex.extract(\"{output_blob}\", out)")

    # 5. Save & cleanup
    os.makedirs("output", exist_ok=True)
    shutil.copy(pnnx_param, "output/toonclip.param")
    shutil.copy(pnnx_bin, "output/toonclip.bin")
    for f in [sim_file, pnnx_param, pnnx_bin]:
        if os.path.exists(f): os.remove(f)
    
    log("=== Conversion Successful ===")

if __name__ == "__main__":
    main()
