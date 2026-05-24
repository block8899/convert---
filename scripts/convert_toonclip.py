# scripts/convert_toonclip.py
"""
ToonClip ONNX → Simplify → NCNN (via PNNX)

Same pipeline as convert_animegan.py:
  1. Simplify ONNX with onnxsim
  2. Convert via PNNX (produces .ncnn.param + .ncnn.bin)
  3. Rename to toonclip.param / toonclip.bin
  4. Verify
"""

import os
import sys
import subprocess
import shutil


def main():
    print("=== ToonClip ONNX → NCNN (via PNNX) ===\n")

    onnx_file = "ToonClip.onnx"
    if not os.path.exists(onnx_file):
        print(f"MISSING: {onnx_file}")
        sys.exit(1)

    print(f"Input: {os.path.getsize(onnx_file) / 1024 / 1024:.1f} MB")

    # 1. Inspect ONNX
    print("\n1. Inspecting ONNX...")
    import onnx
    model = onnx.load(onnx_file)
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    print(f"   Input:  '{input_name}' shape={input_shape}")
    print(f"   Output: '{output_name}' shape={output_shape}")

    # 2. Simplify ONNX
    print("\n2. Simplifying ONNX...")
    sim_file = "toonclip_sim.onnx"
    ret = subprocess.run(
        [sys.executable, "-m", "onnxsim", onnx_file, sim_file],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"   Simplify failed, using original: {ret.stderr[:200]}")
        shutil.copy(onnx_file, sim_file)
    else:
        print(f"   OK: {os.path.getsize(sim_file) / 1024 / 1024:.1f} MB")

    # 3. Convert ONNX → NCNN via PNNX
    print("\n3. Converting via PNNX...")
    pnnx_param = "toonclip_sim.ncnn.param"
    pnnx_bin = "toonclip_sim.ncnn.bin"

    # Clean previous PNNX output if any
    for f in [pnnx_param, pnnx_bin]:
        if os.path.exists(f):
            os.remove(f)

    ret = subprocess.run(
        ["pnnx", sim_file],
        capture_output=True, text=True, timeout=300,
    )
    print(f"   stdout (last 500): {ret.stdout[-500:]}")
    if ret.returncode != 0:
        print(f"   stderr (last 500): {ret.stderr[-500:]}")
        sys.exit(1)

    if not os.path.exists(pnnx_param) or not os.path.exists(pnnx_bin):
        print("   PNNX output files not found!")
        sys.exit(1)

    print(f"   OK: param={os.path.getsize(pnnx_param) / 1024:.1f} KB, "
          f"bin={os.path.getsize(pnnx_bin) / 1024 / 1024:.1f} MB")

    # 4. Copy to output
    os.makedirs("output", exist_ok=True)
    shutil.copy(pnnx_param, "output/toonclip.param")
    shutil.copy(pnnx_bin, "output/toonclip.bin")

    # 5. Verify
    print("\n=== Output ===")
    for f in ["output/toonclip.param", "output/toonclip.bin"]:
        if os.path.exists(f):
            size = os.path.getsize(f)
            if size > 1024 * 1024:
                print(f"  {f}: {size / 1024 / 1024:.1f} MB")
            else:
                print(f"  {f}: {size / 1024:.1f} KB")
        else:
            print(f"  MISSING: {f}")
            sys.exit(1)

    # 6. Print param info
    with open("output/toonclip.param", "r") as f:
        lines = f.readlines()
    print(f"\n  Param lines: {len(lines)}")
    for line in lines:
        if line.startswith("Input"):
            print(f"  Input layer: {line.strip()}")
    print(f"  Last layer: {lines[-1].strip()}")

    print("\nToonClip OK!")


if __name__ == "__main__":
    main()
