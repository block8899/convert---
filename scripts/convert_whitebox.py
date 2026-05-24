# scripts/convert_whitebox.py
"""
White-Box Cartoonization TensorFlow → ONNX → NCNN

Pipeline:
  1. Load TF SavedModel (saved_models/)
  2. Convert TF → ONNX via tf2onnx
  3. Simplify with onnxsim
  4. Convert ONNX → NCNN via PNNX
  5. Verify

White-Box has 3 sub-networks:
  - Edge smoothing (generator_smooth)
  - Structure (generator_structure)
  - Texture (generator_texture)

The main inference uses: texture network + structure network + guided filter.
For mobile, we convert the texture generator (main network).
"""

import os
import sys
import subprocess
import shutil
import glob


def find_saved_model(repo_dir):
    """Find TF SavedModel directory."""
    # Common locations
    candidates = [
        os.path.join(repo_dir, "saved_models"),
        os.path.join(repo_dir, "checkpoints"),
        os.path.join(repo_dir, "model"),
        os.path.join(repo_dir, "pretrained"),
    ]

    for d in candidates:
        if os.path.isdir(d):
            # Check for saved_model.pb
            for root, dirs, files in os.walk(d):
                if "saved_model.pb" in files:
                    return root
                # Check for .pb files (frozen graph)
                pb_files = [f for f in files if f.endswith(".pb")]
                if pb_files:
                    return root

    # Search entire repo
    for root, dirs, files in os.walk(repo_dir):
        if "saved_model.pb" in files:
            return root

    return None


def find_frozen_graphs(repo_dir):
    """Find frozen .pb graph files."""
    pb_files = []
    for root, dirs, files in os.walk(repo_dir):
        for f in files:
            if f.endswith(".pb") and "saved_model" not in f:
                full_path = os.path.join(root, f)
                size_mb = os.path.getsize(full_path) / 1024 / 1024
                if size_mb > 0.1:  # Skip tiny files
                    pb_files.append((full_path, size_mb))
    return pb_files


def find_checkpoint_files(repo_dir):
    """Find TF checkpoint files."""
    ckpt_files = []
    for root, dirs, files in os.walk(repo_dir):
        for f in files:
            if f.endswith((".index", ".data-00000-of-00001")):
                ckpt_files.append(os.path.join(root, f))
    return ckpt_files


def convert_tf_savedmodel_to_onnx(saved_model_dir, output_path):
    """Convert TF SavedModel → ONNX."""
    print(f"  Converting SavedModel: {saved_model_dir}")

    # Get input signature
    import tensorflow as tf
    model = tf.saved_model.load(saved_model_dir)
    sig = model.signatures["serving_default"]
    input_info = sig.structured_input_signature
    print(f"  Input signature: {input_info}")

    # Use tf2onnx
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", saved_model_dir,
        "--output", output_path,
        "--opset", "11",
    ]

    ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(f"  stdout: {ret.stdout[-500:]}")
    if ret.returncode != 0:
        print(f"  stderr: {ret.stderr[-500:]}")
        return False

    return os.path.exists(output_path)


def convert_frozen_pb_to_onnx(pb_path, output_path, input_names, output_names):
    """Convert frozen .pb → ONNX."""
    print(f"  Converting frozen graph: {pb_path}")

    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--graphdef", pb_path,
        "--output", output_path,
        "--opset", "11",
        "--inputs", input_names,
        "--outputs", output_names,
    ]

    ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(f"  stdout: {ret.stdout[-500:]}")
    if ret.returncode != 0:
        print(f"  stderr: {ret.stderr[-500:]}")
        return False

    return os.path.exists(output_path)


def simplify_onnx(input_path, output_path):
    """Simplify ONNX model."""
    import onnx
    from onnxsim import simplify

    model = onnx.load(input_path)
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    print(f"  ONNX input: '{input_name}', output: '{output_name}'")

    model_simp, check = simplify(model)
    if check:
        onnx.save(model_simp, output_path)
        print(f"  Simplified OK: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
        return True, input_name, output_name
    else:
        print("  Simplify failed, using original")
        shutil.copy(input_path, output_path)
        return False, input_name, output_name


def convert_onnx_to_ncnn(onnx_path, param_path, bin_path):
    """Convert ONNX → NCNN via PNNX."""
    print(f"  Converting via PNNX: {onnx_path}")

    # Clean previous PNNX output
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
    print("=== White-Box Cartoonization → NCNN ===\n")

    repo_dir = "repo_wbc"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(repo_dir):
        print(f"MISSING: {repo_dir}/")
        sys.exit(1)

    # 1. Analyze repo structure
    print("1. Analyzing repo structure...")

    saved_model_dir = find_saved_model(repo_dir)
    frozen_graphs = find_frozen_graphs(repo_dir)
    checkpoint_files = find_checkpoint_files(repo_dir)

    print(f"   SavedModel dir: {saved_model_dir}")
    print(f"   Frozen graphs: {[(f, f'{s:.1f}MB') for f, s in frozen_graphs]}")
    print(f"   Checkpoint files: {checkpoint_files}")

    # List all .pb files
    print("\n   All .pb files:")
    for root, dirs, files in os.walk(repo_dir):
        for f in files:
            if f.endswith(".pb"):
                path = os.path.join(root, f)
                size = os.path.getsize(path) / 1024 / 1024
                print(f"     {path} ({size:.1f} MB)")

    # List all .h5 files
    print("\n   All .h5 files:")
    for root, dirs, files in os.walk(repo_dir):
        for f in files:
            if f.endswith(".h5"):
                path = os.path.join(root, f)
                size = os.path.getsize(path) / 1024 / 1024
                print(f"     {path} ({size:.1f} MB)")

    # 2. Try conversion based on what we found
    print("\n2. Converting model...")

    onnx_path = os.path.join(output_dir, "whitebox_sim.onnx")
    param_path = os.path.join(output_dir, "whitebox.param")
    bin_path = os.path.join(output_dir, "whitebox.bin")

    converted = False

    # Strategy 1: SavedModel → ONNX
    if saved_model_dir and not converted:
        print(f"\n   Strategy 1: SavedModel → ONNX")
        raw_onnx = os.path.join(output_dir, "whitebox_raw.onnx")
        if convert_tf_savedmodel_to_onnx(saved_model_dir, raw_onnx):
            ok, inp, outp = simplify_onnx(raw_onnx, onnx_path)
            converted = True

    # Strategy 2: Frozen .pb → ONNX
    if frozen_graphs and not converted:
        print(f"\n   Strategy 2: Frozen .pb → ONNX")
        # Use largest .pb file (likely the main model)
        frozen_graphs.sort(key=lambda x: x[1], reverse=True)
        pb_path, pb_size = frozen_graphs[0]
        print(f"   Using: {pb_path} ({pb_size:.1f} MB)")

        # Need to determine input/output names
        # Try common names
        input_names = "Placeholder:0"
        output_names = "generator/output:0"

        raw_onnx = os.path.join(output_dir, "whitebox_raw.onnx")
        if convert_frozen_pb_to_onnx(pb_path, raw_onnx, input_names, output_names):
            ok, inp, outp = simplify_onnx(raw_onnx, onnx_path)
            converted = True

    if not converted:
        print("\n   FAILED: Could not find convertible model!")
        print("   The repo may need weights downloaded from Google Drive.")
        print("   Check README for download instructions.")
        print("\n   Repo contents:")
        for root, dirs, files in os.walk(repo_dir):
            level = root.replace(repo_dir, "").count(os.sep)
            indent = " " * 2 * level
            print(f"   {indent}{os.path.basename(root)}/")
            if level < 2:
                subindent = " " * 2 * (level + 1)
                for f in files[:10]:
                    fsize = os.path.getsize(os.path.join(root, f))
                    print(f"   {subindent}{f} ({fsize/1024:.0f} KB)")
        sys.exit(1)

    # 3. ONNX → NCNN via PNNX
    print("\n3. Converting ONNX → NCNN...")

    import onnx
    model = onnx.load(onnx_path)
    inp = model.graph.input[0].name
    outp = model.graph.output[0].name

    if not convert_onnx_to_ncnn(onnx_path, param_path, bin_path):
        print("   PNNX conversion failed!")
        sys.exit(1)

    # 4. Fix tensor names
    print("\n4. Fixing tensor names...")
    fix_tensor_names(param_path, inp, outp)

    # 5. Verify
    print("\n=== Output ===")
    verify_model(param_path, bin_path)
    print("\nWhite-Box Cartoonization DONE!")


if __name__ == "__main__":
    main()
