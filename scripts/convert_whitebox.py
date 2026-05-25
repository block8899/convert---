# scripts/convert_whitebox.py
"""
White-Box Cartoonization TF1 -> Frozen PB -> ONNX -> NCNN
"""

import os
import sys
import subprocess
import shutil


def main():
    print("=== White-Box Cartoonization TF1 -> NCNN ===\n")

    repo_dir = "repo_wbc"
    test_code_dir = os.path.join(repo_dir, "test_code")
    ckpt_path = os.path.join(test_code_dir, "saved_models", "model-33999")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(ckpt_path + ".index"):
        print(f"MISSING checkpoint: {ckpt_path}.index")
        sys.exit(1)

    print(f"Checkpoint: {ckpt_path}")
    data_file = ckpt_path + ".data-00000-of-00001"
    print(f"Data size: {os.path.getsize(data_file) / 1024 / 1024:.1f} MB")

    # 1. Load TF1 checkpoint -> frozen .pb
    print("\n1. Loading TF1 model and exporting frozen .pb...")

    frozen_pb = os.path.join(output_dir, "whitebox_frozen.pb")
    export_py = os.path.join(output_dir, "_export_frozen.py")

    export_lines = [
        "import os",
        "import sys",
        "import tensorflow as tf",
        "",
        "# Force TF1 behavior",
        "tf1 = tf.compat.v1",
        "tf1.disable_eager_execution()",
        "",
        "# Replace tensorflow.contrib.slim with standalone tf_slim",
        "import tf_slim as slim",
        "contrib_mock = type(sys)('tensorflow.contrib')",
        "contrib_mock.slim = slim",
        "sys.modules['tensorflow.contrib'] = contrib_mock",
        "sys.modules['tensorflow.contrib.slim'] = slim",
        "",
        "# Re-export TF1 APIs into tf module so network.py can use tf.xxx",
        "tf.variable_scope = tf1.variable_scope",
        "tf.get_variable = tf1.get_variable",
        "tf.placeholder = tf1.placeholder",
        "tf.Session = tf1.Session",
        "tf.train.Saver = tf1.train.Saver",
        "tf.GraphDef = tf1.GraphDef",
        "tf.image.resize_bilinear = tf.compat.v1.image.resize_bilinear",
        "tf.image.resize_nearest_neighbor = tf.compat.v1.image.resize_nearest_neighbor",
        "",
        f"test_code_dir = r'{test_code_dir}'",
        "sys.path.insert(0, test_code_dir)",
        "",
        "import network",
        "unet_generator = network.unet_generator",
        "",
        "print('Building graph...')",
        "input_ph = tf1.placeholder(tf.float32, [1, 512, 512, 3], name='input')",
        "output = unet_generator(input_ph, channel=32, num_blocks=4, name='generator', reuse=False)",
        "output = tf.identity(output, name='output')",
        "print(f'Output shape: {output.shape}')",
        "",
        f"ckpt_path = r'{ckpt_path}'",
        "print(f'Loading checkpoint: {ckpt_path}')",
        "saver = tf1.train.Saver()",
        "sess = tf1.Session()",
        "saver.restore(sess, ckpt_path)",
        "print('Checkpoint loaded OK')",
        "",
        "print('Freezing graph...')",
        "graph_def = tf1.graph_util.convert_variables_to_constants(",
        "    sess, sess.graph_def, ['output'])",
        "",
        f"frozen_path = r'{frozen_pb}'",
        "with tf.io.gfile.GFile(frozen_path, 'wb') as fout:",
        "    fout.write(graph_def.SerializeToString())",
        "print(f'Frozen .pb: {os.path.getsize(frozen_path)/1024/1024:.1f} MB')",
        "",
        "for node in graph_def.node:",
        "    if node.op == 'Placeholder':",
        "        print(f'Input: {node.name}')",
        "    if 'output' in node.name.lower():",
        "        print(f'Output: {node.name} op={node.op}')",
        "",
        "sess.close()",
        "print('Export DONE')",
    ]

    with open(export_py, "w") as f:
        f.write("\n".join(export_lines) + "\n")

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

    # 2. Frozen .pb -> ONNX
    print("\n2. Converting frozen .pb -> ONNX...")

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

    # 4. ONNX -> NCNN via PNNX
    print("\n4. Converting ONNX -> NCNN via PNNX...")

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
