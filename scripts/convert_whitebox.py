# scripts/convert_whitebox.py
"""
White-Box Cartoonization TF1 → Frozen PB → ONNX → NCNN

Pipeline:
  1. Load TF1 checkpoint using repo's network.py architecture
  2. Export to frozen .pb
  3. Convert frozen .pb → ONNX via tf2onnx
  4. Simplify with onnxsim
  5. Convert ONNX → NCNN via PNNX
  6. Fix tensor names to in0/out0

Repo: https://github.com/SystemErrorWang/White-box-Cartoonization
"""

import os
import sys
import subprocess
import shutil


def main():
    print("=== White-Box Cartoonization TF1 → NCNN ===\n")

    repo_dir = "repo_wbc"
    test_code_dir = os.path.join(repo_dir, "test_code")
    ckpt_path = os.path.join(test_code_dir, "saved_models", "model-33999")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Verify checkpoint exists
    if not os.path.exists(ckpt_path + ".index"):
        print(f"MISSING checkpoint: {ckpt_path}.index")
        sys.exit(1)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Data size: {os.path.getsize(ckpt_path + '.data-00000-of-00001') / 1024 / 1024:.1f} MB")

    # ─────────────────────────────────────────────
    # 1. Load TF1 model and export to frozen .pb
    # ─────────────────────────────────────────────
    print("\n1. Loading TF1 model and exporting frozen .pb...")

    # Add test_code to path so we can import network
    sys.path.insert(0, test_code_dir)

    export_script = '''
import os
import sys
import tensorflow as tf

# Force TF1 behavior
tf1 = tf.compat.v1
tf1.disable_eager_execution()

# Add test_code to path
test_code_dir = "{test_code_dir}"
sys.path.insert(0, test_code_dir)

from network import unet_generator

print("Building graph...")

# Build graph — White-Box uses 512x512 input
input_ph = tf1.placeholder(tf.float32, [1, 512, 512, 3], name="input")

# Build generator (same as test_code/cartoonize.py)
output = unet_generator(input_ph, channel=32, num_blocks=4, name="generator", reuse=False)

# Output name
output = tf.identity(output, name="output")
print(f"Output shape: {{output.shape}}")

# Load checkpoint
ckpt_path = "{ckpt_path}"
print(f"Loading checkpoint: {{ckpt_path}}")

saver = tf1.train.Saver()
sess = tf1.Session()
saver.restore(sess, ckpt_path)
print("Checkpoint loaded OK")

# Freeze graph
print("Freezing graph...")
graph_def = tf1.graph_util.convert_variables_to_constants(
    sess,
    sess.graph_def,
    ["output"]
)

# Save frozen .pb
frozen_path = os.path.join("{output_dir}", "whitebox_frozen.pb")
with tf.io.gfile.GFile(frozen_path, "wb") as f:
    f.write(graph_def.SerializeToString())

size = os.path.getsize(frozen_path) / 1024 / 1024
print(f"Frozen .pb saved: {{frozen_path}} ({{size:.1f}} MB)")

# Print tensor names for verification
for node in graph_def.node:
    if node.op == "Placeholder":
        print(f"Input: {{node.name}}")
    if "output" in node.name.lower() and node.op in ("Identity", "Conv2D", "BiasAdd"):
        print(f"Output candidate: {{node.name}} (op={{node.op}})")

sess.close()
print("Export DONE")
'''.format(
        test_code_dir=test_code_dir,
        ckpt_path=ckpt_path,
        output_dir=output_dir,
    )

    export_py = os.path.join(output_dir, "_export_frozen.py")
    with open(export_py, 'w') as f:
        f.write(export_script)

    ret = subprocess.run(
        [sys.executable, export_py],
        capture_output=True, text=True, timeout=300,
    )
    print(f"stdout:\n{ret.stdout}")
    if ret.returncode != 0:
        print(f"stderr:\n{ret.stderr}")
        sys.exit(1)

    frozen_pb = os.path.join(output_dir, "whitebox_frozen.pb")
    if not os.path.exists(frozen_pb):
        print("Frozen .pb not created!")
        sys.exit(1)

    print(f"Frozen .pb: {os.path.getsize(frozen_pb) / 1024 / 1024:.1f} MB")

    # ─────────────────────────────────────────────
    # 2. Convert frozen .pb → ONNX
    # ─────────────────────────────────────────────
    print("\n2. Converting frozen .pb → ONNX...")

    raw_onnx = os.path.join(output_dir, "whitebox_raw.onnx")

    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--graphdef", frozen_pb,
        "--output", raw_onnx,
        "--opset", "11",
        "--inputs", "input:0",
        "--outputs", "output:0",
    ]

    ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(f"stdout:\n{ret.stdout[-500:]}")
    if ret.returncode != 0:
        print(f"stderr:\n{ret.stderr[-500:]}")
        # Try alternative output name
        print("Trying alternative output name...")
        cmd[cmd.index("--outputs") + 1] = "output:0"
        for node_name in ["generator/output:0", "generator/G_conv9/BiasAdd:0",
                          "generator/G_conv9/Conv2D:0", "output_1:0"]:
            cmd[cmd.index("--outputs") + 1] = node_name
            print(f"  Trying: {node_name}")
            ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if ret.returncode == 0 and os.path.exists(raw_onnx):
                print(f"  Success with: {node_name}")
                break
            print(f"  Failed: {ret.stderr[-200:]}")

    if not os.path.exists(raw_onnx):
        print("ONNX conversion failed!")
        # Debug: list all ops in frozen graph
        print("\nDebugging frozen graph ops...")
        debug_script = '''
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
with tf.io.gfile.GFile("{pb}", "rb") as f:
    gd = tf.compat.v1.GraphDef()
    gd.ParseFromString(f.read())
for n in gd.node:
    if "output" in n.name.lower() or n.op == "Placeholder":
        print(f"  {{n.op}}: {{n.name}}")
'''.format(pb=frozen_pb)
        debug_py = os.path.join(output_dir, "_debug.py")
        with open(debug_py, 'w') as f:
            f.write(debug_script)
        subprocess.run([sys.executable, debug_py], timeout=60)
        sys.exit(1)

    print(f"Raw ONNX: {os.path.getsize(raw_onnx) / 1024 / 1024:.1f} MB")

    # ─────────────────────────────────────────────
    # 3. Simplify ONNX
    # ─────────────────────────────────────────────
    print("\n3. Simplifying ONNX...")

    import onnx
    from onnxsim import simplify

    model = onnx.load(raw_onnx)
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    print(f"  Input:  '{input_name}' shape={input_shape}")
    print(f"  Output: '{output_name}' shape={output_shape}")

    sim_path = os.path.join(output_dir, "whitebox_sim.onnx")
    model_simp, check = simplify(model)
    if check:
        onnx.save(model_simp, sim_path)
        print(f"  Simplified: {os.path.getsize(sim_path) / 1024 / 1024:.1f} MB")
    else:
        print("  Simplify failed, using original")
        shutil.copy(raw_onnx, sim_path)

    # ─────────────────────────────────────────────
    # 4. Convert ONNX → NCNN via PNNX
    # ─────────────────────────────────────────────
    print("\n4. Converting ONNX → NCNN via PNNX...")

    pnnx_param = sim_path.replace(".onnx", ".ncnn.param")
    pnnx_bin = sim_path.replace(".onnx", ".ncnn.bin")

    for f in [pnnx_param, pnnx_bin]:
        if os.path.exists(f):
            os.remove(f)

    ret = subprocess.run(
        ["pnnx", sim_path],
        capture_output=True, text=True, timeout=300,
    )
    print(f"  stdout (last 500): {ret.stdout[-500:]}")
    if ret.returncode != 0:
        print(f"  stderr (last 500): {ret.stderr[-500:]}")
        sys.exit(1)

    if not os.path.exists(pnnx_param) or not os.path.exists(pnnx_bin):
        print("  PNNX output not found!")
        sys.exit(1)

    param_path = os.path.join(output_dir, "whitebox.param")
    bin_path = os.path.join(output_dir, "whitebox.bin")
    shutil.copy(pnnx_param, param_path)
    shutil.copy(pnnx_bin, bin_path)
    print(f"  param: {os.path.getsize(param_path) / 1024:.1f} KB")
    print(f"  bin:   {os.path.getsize(bin_path) / 1024 / 1024:.1f} MB")

    # ─────────────────────────────────────────────
    # 5. Fix tensor names
    # ─────────────────────────────────────────────
    print("\n5. Fixing tensor names...")

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

    # ─────────────────────────────────────────────
    # 6. Verify
    # ─────────────────────────────────────────────
    print("\n=== Output ===")
    print(f"  whitebox.param: {os.path.getsize(param_path) / 1024:.1f} KB, "
          f"{os.path.getsize(bin_path) / 1024 / 1024:.1f} MB")

    with open(param_path, 'r') as f:
        lines = f.readlines()
    print(f"  Lines: {len(lines)}")
    for line in lines:
        if line.startswith('Input'):
            print(f"  Input: {line.strip()}")
    print(f"  Last: {lines[-1].strip()}")

    print("\nWhite-Box Cartoonization DONE!")


if __name__ == "__main__":
    main()
