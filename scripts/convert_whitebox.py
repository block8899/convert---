# scripts/convert_whitebox.py
"""
White-Box Cartoonization TF → ONNX → NCNN

Pipeline (same pattern as convert_animegan.py):
  1. Find TF model (SavedModel / frozen .pb / checkpoint)
  2. Convert TF → ONNX via tf2onnx
  3. Simplify with onnxsim
  4. Convert ONNX → NCNN via PNNX
  5. Fix tensor names to in0/out0
  6. Verify

White-Box Cartoonization repo:
  https://github.com/SystemErrorWang/White-box-Cartoonization

The texture generator is the main inference network.
"""

import os
import sys
import subprocess
import shutil
import glob


def find_tf_model(repo_dir):
    """Find TF model — SavedModel, frozen .pb, or checkpoint."""
    results = {
        'saved_model': None,
        'frozen_pb': [],
        'checkpoints': [],
    }

    for root, dirs, files in os.walk(repo_dir):
        for f in files:
            path = os.path.join(root, f)

            if f == 'saved_model.pb':
                results['saved_model'] = root

            elif f.endswith('.pb') and 'saved_model' not in f:
                size = os.path.getsize(path) / 1024 / 1024
                if size > 0.1:
                    results['frozen_pb'].append((path, size))

            elif f.endswith('.index'):
                ckpt = path.replace('.index', '')
                results['checkpoints'].append(ckpt)

    return results


def inspect_frozen_pb(pb_path):
    """Inspect frozen .pb to find input/output tensor names."""
    import tensorflow as tf

    print(f"  Inspecting: {pb_path}")

    with tf.io.gfile.GFile(pb_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # List all ops
    inputs = []
    outputs = []
    all_ops = []

    for node in graph_def.node:
        all_ops.append(f"{node.op}: {node.name}")

        # Find Placeholder inputs
        if node.op == 'Placeholder':
            dtype = node.attr.get('dtype', {})
            shape = node.attr.get('shape', {})
            inputs.append(node.name)

    # Find last Conv2D or output nodes (common output patterns)
    for node in graph_def.node:
        name = node.name.lower()
        if any(k in name for k in ['output', 'generator', 'tanh', 'sigmoid', 'final']):
            if node.op in ('Conv2D', 'Tanh', 'Sigmoid', 'Mul', 'Add', 'BiasAdd'):
                outputs.append(node.name)

    print(f"  Ops count: {len(all_ops)}")
    print(f"  Inputs: {inputs}")
    print(f"  Candidate outputs: {outputs}")

    # Show first 10 and last 10 ops
    print(f"  First 10 ops:")
    for op in all_ops[:10]:
        print(f"    {op}")
    print(f"  Last 10 ops:")
    for op in all_ops[-10:]:
        print(f"    {op}")

    return inputs, outputs


def convert_savedmodel_to_onnx(saved_model_dir, output_path):
    """Convert TF SavedModel → ONNX."""
    print(f"  Converting SavedModel: {saved_model_dir}")

    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", saved_model_dir,
        "--output", output_path,
        "--opset", "11",
    ]

    ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(f"  stdout: {ret.stdout[-300:]}")
    if ret.returncode != 0:
        print(f"  stderr: {ret.stderr[-300:]}")
        return False
    return os.path.exists(output_path)


def convert_frozen_pb_to_onnx(pb_path, output_path, input_names, output_names):
    """Convert frozen .pb → ONNX."""
    print(f"  Converting frozen .pb: {pb_path}")
    print(f"  Inputs: {input_names}")
    print(f"  Outputs: {output_names}")

    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--graphdef", pb_path,
        "--output", output_path,
        "--opset", "11",
        "--inputs", input_names,
        "--outputs", output_names,
    ]

    ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(f"  stdout: {ret.stdout[-300:]}")
    if ret.returncode != 0:
        print(f"  stderr: {ret.stderr[-300:]}")
        return False
    return os.path.exists(output_path)


def convert_checkpoint_to_onnx(repo_dir, ckpt_path, output_path):
    """Convert TF checkpoint → ONNX via SavedModel intermediate."""
    print(f"  Converting checkpoint: {ckpt_path}")

    # Try using the repo's own inference code to export
    test_code_dir = os.path.join(repo_dir, "test_code")
    network_py = os.path.join(test_code_dir, "network.py")

    if os.path.exists(network_py):
        print(f"  Found network.py at: {network_py}")
        # Read network to understand architecture
        with open(network_py, 'r') as f:
            print(f"  Network code preview:")
            for i, line in enumerate(f.readlines()[:30]):
                print(f"    {line.rstrip()}")

    # Try tf2onnx with checkpoint
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--checkpoint", ckpt_path + ".index",
        "--output", output_path,
        "--opset", "11",
        "--inputs", "input:0",
        "--outputs", "output:0",
    ]

    ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(f"  stdout: {ret.stdout[-300:]}")
    if ret.returncode != 0:
        print(f"  stderr: {ret.stderr[-300:]}")
        return False
    return os.path.exists(output_path)


def simplify_onnx(input_path, output_path):
    """Simplify ONNX."""
    import onnx
    from onnxsim import simplify

    model = onnx.load(input_path)
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    print(f"  ONNX input: '{input_name}' shape={input_shape}")
    print(f"  ONNX output: '{output_name}' shape={output_shape}")

    model_simp, check = simplify(model)
    if check:
        onnx.save(model_simp, output_path)
        print(f"  Simplified OK")
    else:
        print("  Simplify failed, using original")
        shutil.copy(input_path, output_path)

    return input_name, output_name


def convert_onnx_to_ncnn(onnx_path, param_path, bin_path):
    """Convert ONNX → NCNN via PNNX."""
    base = os.path.splitext(onnx_path)[0]
    pnnx_param = base + ".ncnn.param"
    pnnx_bin = base + ".ncnn.bin"

    for f in [pnnx_param, pnnx_bin]:
        if os.path.exists(f):
            os.remove(f)

    ret = subprocess.run(
        ["pnnx", onnx_path],
        capture_output=True, text=True, timeout=300,
    )
    print(f"  stdout (last 500): {ret.stdout[-500:]}")

    if ret.returncode != 0:
        print(f"  stderr (last 500): {ret.stderr[-500:]}")
        return False

    if not os.path.exists(pnnx_param) or not os.path.exists(pnnx_bin):
        print("  PNNX output not found!")
        return False

    shutil.copy(pnnx_param, param_path)
    shutil.copy(pnnx_bin, bin_path)
    return True


def fix_tensor_names(param_path, old_input, old_output):
    """Rename tensor names to in0/out0."""
    with open(param_path, 'r') as f:
        content = f.read()

    if old_input != 'in0':
        content = content.replace(f' {old_input} ', ' in0 ')
        content = content.replace(f' {old_input}\n', ' in0\n')
    if old_output != 'out0':
        content = content.replace(f' {old_output} ', ' out0 ')
        content = content.replace(f' {old_output}\n', ' out0\n')

    with open(param_path, 'w') as f:
        f.write(content)


def verify_model(param_path, bin_path):
    """Print model info."""
    print(f"\n  {os.path.basename(param_path)}: "
          f"{os.path.getsize(param_path) / 1024:.1f} KB, "
          f"{os.path.getsize(bin_path) / 1024 / 1024:.1f} MB")

    with open(param_path, 'r') as f:
        lines = f.readlines()

    print(f"  Lines: {len(lines)}")
    for line in lines:
        if line.startswith('Input'):
            print(f"  Input: {line.strip()}")
    print(f"  Last: {lines[-1].strip()}")


def main():
    print("=== White-Box Cartoonization TF → NCNN ===\n")

    repo_dir = "repo_wbc"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(repo_dir):
        print(f"MISSING: {repo_dir}/")
        sys.exit(1)

    # 1. Find TF model
    print("1. Finding TF model...")
    models = find_tf_model(repo_dir)
    print(f"   SavedModel: {models['saved_model']}")
    print(f"   Frozen .pb: {[(f, f'{s:.1f}MB') for f, s in models['frozen_pb']]}")
    print(f"   Checkpoints: {models['checkpoints']}")

    # 2. Convert TF → ONNX
    print("\n2. Converting TF → ONNX...")
    raw_onnx = os.path.join(output_dir, "whitebox_raw.onnx")
    onnx_path = os.path.join(output_dir, "whitebox_sim.onnx")
    param_path = os.path.join(output_dir, "whitebox.param")
    bin_path = os.path.join(output_dir, "whitebox.bin")

    converted = False

    # Strategy 1: SavedModel
    if models['saved_model'] and not converted:
        print("\n   Strategy 1: SavedModel → ONNX")
        if convert_savedmodel_to_onnx(models['saved_model'], raw_onnx):
            converted = True

    # Strategy 2: Frozen .pb
    if models['frozen_pb'] and not converted:
        print("\n   Strategy 2: Frozen .pb → ONNX")
        models['frozen_pb'].sort(key=lambda x: x[1], reverse=True)
        pb_path, pb_size = models['frozen_pb'][0]
        print(f"   Using: {pb_path} ({pb_size:.1f} MB)")

        inputs, outputs = inspect_frozen_pb(pb_path)

        if inputs and outputs:
            input_names = inputs[0] + ":0"
            output_names = outputs[-1] + ":0"
            if convert_frozen_pb_to_onnx(pb_path, raw_onnx, input_names, output_names):
                converted = True
        else:
            print("   Could not determine input/output names!")
            # Try common names
            for inp in ["input:0", "Placeholder:0", "input_image:0"]:
                for out in ["output:0", "generator/output:0", "generator/G_conv9/BiasAdd:0"]:
                    print(f"   Trying: {inp} → {out}")
                    if convert_frozen_pb_to_onnx(pb_path, raw_onnx, inp, out):
                        converted = True
                        break
                if converted:
                    break

    # Strategy 3: Checkpoint
    if models['checkpoints'] and not converted:
        print("\n   Strategy 3: Checkpoint → ONNX")
        ckpt = models['checkpoints'][0]
        print(f"   Using: {ckpt}")
        if convert_checkpoint_to_onnx(repo_dir, ckpt, raw_onnx):
            converted = True

    if not converted:
        print("\n   FAILED: Could not convert any TF model!")
        print("   Dumping repo structure for debug:")
        for root, dirs, files in os.walk(repo_dir):
            level = root.replace(repo_dir, '').count(os.sep)
            if level > 3:
                continue
            indent = '  ' * level
            print(f"   {indent}{os.path.basename(root)}/")
            subindent = '  ' * (level + 1)
            for f in sorted(files)[:15]:
                fsize = os.path.getsize(os.path.join(root, f))
                print(f"   {subindent}{f} ({fsize/1024:.0f} KB)")
        sys.exit(1)

    # 3. Simplify ONNX
    print("\n3. Simplifying ONNX...")
    input_name, output_name = simplify_onnx(raw_onnx, onnx_path)

    # 4. ONNX → NCNN via PNNX
    print("\n4. Converting ONNX → NCNN...")
    if not convert_onnx_to_ncnn(onnx_path, param_path, bin_path):
        print("   PNNX conversion failed!")
        sys.exit(1)

    # 5. Fix tensor names
    print("\n5. Fixing tensor names...")
    fix_tensor_names(param_path, input_name, output_name)

    # 6. Verify
    print("\n=== Output ===")
    verify_model(param_path, bin_path)
    print("\nWhite-Box Cartoonization DONE!")


if __name__ == "__main__":
    main()
