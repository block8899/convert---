"""
ToonClip ONNX → NCNN (via PNNX) - Production Ready
✅ Fixed: inputshape, fp16, device control, blob extraction, cleanup
"""
import os
import sys
import subprocess
import shutil
import onnx

def log(msg, level="INFO"):
    print(f"[{level}] {msg}")

def main():
    log("=== ToonClip ONNX → NCNN (via PNNX) ===")

    onnx_file = "ToonClip.onnx"
    if not os.path.exists(onnx_file):
        log(f"MISSING: {onnx_file}", "ERROR")
        sys.exit(1)

    log(f"Input model: {onnx_file} ({os.path.getsize(onnx_file) / 1024 / 1024:.2f} MB)")

    # 1. Inspect ONNX
    log("1. Inspecting ONNX...")
    model = onnx.load(onnx_file)
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    in_shape = [d.dim_value if d.dim_value else -1 for d in model.graph.input[0].type.tensor_type.shape.dim]
    out_shape = [d.dim_value if d.dim_value else -1 for d in model.graph.output[0].type.tensor_type.shape.dim]
    log(f"   ONNX Input:  '{in_name}' shape={in_shape}")
    log(f"   ONNX Output: '{out_name}' shape={out_shape}")

    # 2. Simplify ONNX
    log("2. Simplifying ONNX...")
    sim_file = "toonclip_sim.onnx"
    try:
        subprocess.run([sys.executable, "-m", "onnxsim", onnx_file, sim_file], check=True, capture_output=True, text=True)
        log(f"   Simplified: {os.path.getsize(sim_file) / 1024 / 1024:.2f} MB")
    except subprocess.CalledProcessError as e:
        log(f"onnxsim failed, using original. Err: {e.stderr[:150]}", "WARN")
        shutil.copy(onnx_file, sim_file)

    # 3. Convert via PNNX
    log("3. Converting via PNNX...")
    pnnx_cmd = [
        "pnnx", sim_file,
        "inputshape=[1,3,128,128]",  # ✅ BẮT BUỘC: Resolve tensor shape cho UNet++
        "device=cpu",                # ✅ Đảm bảo op tương thích Android CPU/Vulkan
        "fp16=0",                    # ✅ Tắt FP16 lúc convert, xử lý precision ở runtime
        "optlevel=2"                 # ✅ Tối ưu an toàn, không phá graph
    ]

    try:
        res = subprocess.run(pnnx_cmd, capture_output=True, text=True, timeout=300)
        if res.stdout:
            log(f"   PNNX stdout (tail):\n{res.stdout[-400:]}")
        if res.returncode != 0:
            log(f"PNNX FAILED!\n{res.stderr[-800:]}", "ERROR")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        log("PNNX timed out (5m)", "ERROR")
        sys.exit(1)

    pnnx_param = "toonclip_sim.ncnn.param"
    pnnx_bin = "toonclip_sim.ncnn.bin"

    if not os.path.exists(pnnx_param) or not os.path.exists(pnnx_bin):
        log("PNNX output missing!", "ERROR")
        sys.exit(1)

    log(f"   Converted: param={os.path.getsize(pnnx_param)/1024:.1f} KB, bin={os.path.getsize(pnnx_bin)/1024/1024:.2f} MB")

    # 4. Extract NCNN blob names
    log("4. Detecting NCNN blob names...")
    input_blob = None
    output_blob = None
    with open(pnnx_param, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    
    # NCNN format: LayerType layer_name input_cnt output_cnt blob_in... blob_out...
    for line in lines:
        if line.startswith("Input"):
            input_blob = line.split()[-1]
        # Last layer is usually output
        output_blob = line.split()[-1]

    if input_blob and output_blob:
        log(f"   ✅ NCNN Input Blob:  '{input_blob}'")
        log(f"   ✅ NCNN Output Blob: '{output_blob}'")
        log("   👉 UPDATE C++ CODE: ex.input(\"<input_blob>\", in) / ex.extract(\"<output_blob>\", out)")
    else:
        log("   ⚠️ Blob auto-detect failed. Open .param in Netron to verify.", "WARN")

    # 5. Save & Cleanup
    os.makedirs("output", exist_ok=True)
    shutil.copy(pnnx_param, "output/toonclip.param")
    shutil.copy(pnnx_bin, "output/toonclip.bin")
    log("   ✅ Saved to output/")

    for f in [sim_file, pnnx_param, pnnx_bin]:
        if os.path.exists(f): os.remove(f)
    log("   🧹 Temp files cleaned.")
    log("=== Conversion Successful ===")

if __name__ == "__main__":
    main()
