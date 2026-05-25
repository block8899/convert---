# scripts/convert_whitebox.py
"""
White-Box Cartoonization TF1 → Frozen PB → ONNX → NCNN
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

    if not os.path.exists(ckpt_path + ".index"):
        print(f"MISSING checkpoint: {ckpt_path}.index")
        sys.exit(1)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Data size: {os.path.getsize(ckpt_path + '.data-00000-of-00001') / 1024 / 1024:.1f} MB")

    # 1. Write export script to file (avoids indentation issues)
    print("\n1. Loading TF1 model and exporting frozen .pb...")

    export_py = os.path.join(output_dir, "_export_frozen.py")
    frozen_pb = os.path.join(output_dir, "whitebox_frozen.pb")

    with open(export_py, "w") as f:
        f.write("import os\n")
        f.write("import sys\n")
        f.write("import tensorflow as tf\n")
        f.write("import importlib.util\n")
        f.write("\n")
        f.write("tf1 = tf.compat.v1\n")
        f.write("tf1.disable_eager_execution()\n")
        f.write("\n")
        f.write("import tf_slim as slim\n")
        f.write("tf.contrib = type(sys)('contrib')\n")
        f.write("tf.contrib.slim = slim\n")
        f.write("\n")
        f.write(f"test_code_dir = r'{test_code_dir}'\n")
        f.write("sys.path.insert(0, test_code_dir)\n")
        f.write("\n")
        f.write("spec = importlib.util.spec_from_file_location(\n")
        f.write("    'network', os.path.join(test_code_dir, 'network.py'))\n")
        f.write("network_mod = importlib.util.module_from_spec(spec)\n")
        f.write("spec.loader.exec_module(network_mod)\n")
        f.write("unet_generator = network_mod.unet_generator\n")
        f.write("\n")
        f.write("print('Building graph...')\n")
        f.write("input_ph = tf1.placeholder(tf.float32, [1, 512, 512, 3], name='input')\n")
        f.write("output = unet_generator(input_ph, channel=32, num_blocks=4, name='generator', reuse=False)\n")
        f.write("output = tf.identity(output, name='output')\n")
        f.write("print(f'Output shape: {output.shape}')\n")
        f.write("\n")
        f.write(f"ckpt_path = r'{ckpt_path}'\n")
        f.write("print(f'Loading checkpoint: {ckpt_path}')\n")
        f.write("saver = tf1.train.Saver()\n")
        f.write("sess = tf1.Session()\n")
        f.write("saver.restore(sess, ckpt_path)\n")
        f.write("print('Checkpoint loaded OK')\n")
        f.write("\n")
        f.write("print('Freezing graph...')\n")
        f.write("graph_def = tf1.graph_util.convert_variables_to_constants(\n")
        f.write("    sess, sess.graph_def, ['output'])\n")
        f.write("\n")
        f.write(f"frozen_path = r'{frozen_pb}'\n")
        f.write("with tf.io.gfile.GFile(frozen_path, 'wb') as fout:\n")
        f.write("    fout.write(graph_def.SerializeToString())\n")
        f.write("print(f'Frozen .pb: {os.path.getsize(frozen_path)/1024/1024:.1f} MB')\n")
        f.write("\n")
        f.write("for node in graph_def.node:\n")
        f.write("    if node.op == 'Placeholder':\n")
        f.write("        print(f'Input: {node.name}')\n")
        f.write("    if 'output' in node.name.lower():\n")
        f.write("        print(f'Output: {node.name} op={node.op}')\n")
        f.write("\n")
        f.write("sess.close()\n")
        f.write("print('Export DONE')\n")

    ret = subprocess.run(
        [sys.executable, export_py],
        capture_output=True, text=True, timeout=300,
    )
    print(f"stdout:\n{ret.stdout}")
    if ret.returncode != 0:
        print(f"stderr:\n{ret.stderr}")
        sys.exit(1)

    if not os.path.exists(frozen_pb):
        print("Frozen .pb not created!")
        sys.exit(1)

    print(f"Frozen .pb: {os.path.getsize(frozen_pb) / 1024 / 1024:.1f} MB")

    # 2. Convert frozen .pb → ONNX
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
        # Try alternative output names
        alt_names = [
            "generator/output:0",
            "generator/G_conv9/BiasAdd:0",
            "generator/G_conv9/Conv2D:0",
            "output_1:0",
        ]
        for name in alt_names:
            cmd[cmd.index("--outputs") + 1] = name
            print(f"  Trying: {name}")
            ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if ret.returncode == 0 and os.path.exists(raw_onnx):
                print(f"  Success: {name}")
                break
            print(f"  Failed: {ret.stderr[-200:]}")

    if not os.path.exists(raw_onnx):
        print("ONNX conversion failed!")
        sys.exit(1)

    print(f"Raw ONNX: {os.path.getsize(raw_onnx) / 1024 / 1024:.1f} MB")

    # 3. Simplify ONNX
    print("\n3. Simplifying ONNX...")

    import onnx
    from onnxsim import simplify

    model = onnx.load(raw_onnx)
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    print(f"  Input:  '{input_name}'")
    print(f"  Output: '{output_name}'")

    sim_path = os.path.join(output_dir, "whitebox_sim.onnx")
    model_simp, check = simplify(model)
    if check:
        onnx.save(model_simp, sim_path)
        print(f"  Simplified: {os.path.getsize(sim_path) / 1024 / 1024:.1f} MB")
    else:
        print("  Simplify failed, using original")
        shutil.copy(raw_onnx, sim_path)

    # 4. Convert ONNX → NCNN via PNNX
    print("\n4. Converting ONNX → NCNN via PNNX...")

    pnnx_param = sim_path.replace(".onnx", ".ncnn.param")
    pnnx_bin = sim_path.replace(".onnx", ".ncnn.bin")

    for fp in [pnnx_param, pnnx_bin]:
        if os.path.exists(fp):
            os.remove(fp)

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

    # 5. Fix tensor names
    print("\n5. Fixing tensor names...")

    with open(param_path, "r") as f:
        content = f.read()

    if input_name != "in0":
        content = content.replace(f" {input_name} ", " in0 ")
        content = content.replace(f" {input_name}\n", " in0\n")
    if output_name != "out0":
        content = content.replace(f" {output_name} ", " out0 ")
        content = content.replace(f" {output_name}\n", " out0\n")

    with open(param_path, "w") as f:
        f.write(content)

    # 6. Verify
    print("\n=== Output ===")
    print(f"  whitebox.param: {os.path.getsize(param_path) / 1024:.1f} KB")
    print(f"  whitebox.bin:   {os.path.getsize(bin_path) / 1024 / 1024:.1f} MB")

    with open(param_path, "r") as f:
        lines = f.readlines()
    print(f"  Lines: {len(lines)}")
    for line in lines:
        if line.startswith("Input"):
            print(f"  Input: {line.strip()}")
    print(f"  Last: {lines[-1].strip()}")

    print("\nWhite-Box Cartoonization DONE!")


if __name__ == "__main__":
    main()
