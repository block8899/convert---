# scripts/convert_toonclip.py
"""
ToonClip ONNX → Simplify → NCNN
"""

import argparse
import os
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True, help='Path to ToonClip .onnx')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--onnx2ncnn', required=True, help='Path to onnx2ncnn binary')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Inspect ONNX
    print("=== Loading ToonClip ONNX ===")
    import onnx
    model = onnx.load(args.onnx)
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    print(f"Input:  '{input_name}' shape={input_shape}")
    print(f"Output: '{output_name}' shape={output_shape}")

    # 2. Simplify
    print("=== Simplifying ONNX ===")
    try:
        from onnxsim import simplify
        model_simp, check = simplify(model)
        if check:
            simp_path = os.path.join(args.output_dir, 'toonclip_sim.onnx')
            onnx.save(model_simp, simp_path)
            print("Simplified OK")
        else:
            print("Simplify check failed, using original")
            simp_path = args.onnx
    except Exception as e:
        print(f"Simplify error: {e}, using original")
        simp_path = args.onnx

    # 3. Convert ONNX → NCNN
    print("=== Converting to NCNN ===")
    param_path = os.path.join(args.output_dir, 'toonclip.param')
    bin_path = os.path.join(args.output_dir, 'toonclip.bin')

    result = subprocess.run(
        [args.onnx2ncnn, simp_path, param_path, bin_path],
        capture_output=True, text=True,
    )
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    if result.returncode != 0:
        print(f"onnx2ncnn failed with code {result.returncode}")
        sys.exit(1)

    # 4. Fix tensor names to in0/out0
    print("=== Fixing tensor names ===")
    with open(param_path, 'r') as f:
        content = f.read()

    if input_name != 'in0':
        content = content.replace(f' {input_name} ', ' in0 ')
        content = content.replace(f' {input_name}\n', ' in0\n')
    if output_name != 'out0':
        content = content.replace(f' {output_name} ', ' out0 ')
        content = content.replace(f' {output_name}\n', ' out0\n')

    with open(param_path, 'w') as f:
        f.write(content)

    # 5. Verify
    print("=== Verifying ===")
    print(f"toonclip.param: {os.path.getsize(param_path)} bytes")
    print(f"toonclip.bin:   {os.path.getsize(bin_path)} bytes")

    with open(param_path, 'r') as f:
        lines = f.readlines()
    print(f"Header: {lines[0].strip()}")
    print(f"Total lines: {len(lines)}")

    for line in lines:
        if line.startswith('Input'):
            print(f"Input layer: {line.strip()}")
    print(f"Last layer: {lines[-1].strip()}")

    print("=== ToonClip conversion DONE ===")


if __name__ == '__main__':
    main()
