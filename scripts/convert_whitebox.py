"""White-Box Cartoonization: TF1 → Frozen PB → ONNX → NCNN (Robust)"""
import os, sys, subprocess, shutil, onnx
from onnxsim import simplify

def run_with_log(cmd, timeout, desc):
    print(f"→ {desc}")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.stdout: print(f"  stdout: {res.stdout[-300:]}")
    if res.returncode != 0:
        print(f"❌ {desc} failed:\n{res.stderr[-1000:]}")
        return False
    return True

def main():
    repo_dir = "repo_wbc"
    test_code_dir = os.path.join(repo_dir, "test_code")
    ckpt_path = os.path.join(test_code_dir, "saved_models", "model-33999")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 0. Validate Checkpoint (CRITICAL: gdown often downloads HTML captchas)
    data_file = ckpt_path + ".data-00000-of-00001"
    if not os.path.exists(data_file):
        print(f"❌ Missing checkpoint data: {data_file}")
        sys.exit(1)
    data_size = os.path.getsize(data_file)
    if data_size < 10 * 1024 * 1024:
        print(f"❌ Checkpoint too small ({data_size/1024:.1f} KB). Likely downloaded as HTML/captcha.")
        print("   Fix: Manually download checkpoint from original repo and place in repo_wbc/test_code/saved_models/")
        sys.exit(1)
    print(f"✅ Checkpoint valid: {data_size/1024/1024:.1f} MB")

    # 1. Export Frozen PB
    frozen_pb = os.path.join(output_dir, "whitebox_frozen.pb")
    export_py = os.path.join(output_dir, "_export_frozen.py")
    
    export_code = f'''
import os, sys, traceback
import tensorflow as tf
tf1 = tf.compat.v1
tf1.disable_eager_execution()

# Mock tf.contrib.slim
import tf_slim as slim
tf.contrib = type(sys)("tensorflow.contrib")
tf.contrib.slim = slim
sys.modules["tensorflow.contrib"] = tf.contrib
sys.modules["tensorflow.contrib.slim"] = slim

# Re-export TF1 APIs
tf.variable_scope = tf1.variable_scope
tf.get_variable = tf1.get_variable
tf.placeholder = tf1.placeholder
tf.Session = tf1.Session
tf.GraphDef = tf1.GraphDef
tf.train.Saver = tf1.train.Saver
tf.image.resize_bilinear = tf1.image.resize_bilinear
tf.image.resize_nearest_neighbor = tf1.image.resize_nearest_neighbor
tf.layers = tf1.layers
tf.nn.conv2d = tf1.nn.conv2d
tf.nn.relu = tf1.nn.relu
tf.nn.bias_add = tf1.nn.bias_add
tf.nn.batch_normalization = tf1.nn.batch_normalization

sys.path.insert(0, r"{test_code_dir}")
import network

try:
    print("Building graph...", flush=True)
    input_ph = tf1.placeholder(tf.float32, [1, 512, 512, 3], name="input")
    output = network.unet_generator(input_ph, channel=32, num_blocks=4, name="generator", reuse=False)
    output = tf.identity(output, name="output")
    print(f"Graph built. Output shape: {{output.shape}}", flush=True)

    print("Loading checkpoint...", flush=True)
    saver = tf1.train.Saver()
    sess = tf1.Session()
    saver.restore(sess, r"{ckpt_path}")
    print("Checkpoint restored.", flush=True)

    print("Freezing graph...", flush=True)
    graph_def = tf1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
    frozen_path = r"{frozen_pb}"
    with tf.io.gfile.GFile(frozen_path, "wb") as f:
        f.write(graph_def.SerializeToString())

    actual_size = os.path.getsize(frozen_path)
    print(f"✅ Frozen PB saved: {{actual_size/1024/1024:.2f}} MB", flush=True)
    if actual_size < 5 * 1024 * 1024:
        print("⚠️ WARNING: PB <5MB. Check if output node name matches graph.", flush=True)

    for node in graph_def.node:
        if "output" in node.name.lower():
            print(f"OUTPUT_NODE: {{node.name}} (op={{node.op}})", flush=True)
    sess.close()
    print("Export DONE", flush=True)
except Exception as e:
    print(f"❌ FATAL: {{e}}", flush=True)
    traceback.print_exc()
    sys.exit(1)
'''
    with open(export_py, "w") as f:
        f.write(export_code)
    
    if not run_with_log([sys.executable, export_py], 300, "Export Frozen PB"):
        sys.exit(1)
    if not os.path.exists(frozen_pb):
        print("❌ Frozen PB file not created"); sys.exit(1)

    # 2. Frozen PB → ONNX
    raw_onnx = os.path.join(output_dir, "whitebox_raw.onnx")
    success = False
    # Try multiple output name patterns
    for out_name in ["output:0", "generator/output:0", "output_1:0", "Identity:0"]:
        cmd = [sys.executable, "-m", "tf2onnx.convert", "--graphdef", frozen_pb, 
               "--output", raw_onnx, "--opset", "11", "--inputs", "input:0", "--outputs", out_name]
        if run_with_log(cmd, 300, f"tf2onnx ({out_name})"):
            if os.path.getsize(raw_onnx) > 5*1024*1024:
                success = True
                break
    if not success:
        print("❌ ONNX conversion failed after trying all output names"); sys.exit(1)

    # 3. Simplify ONNX
    model = onnx.load(raw_onnx)
    onnx_input_name = model.graph.input[0].name
    onnx_output_name = model.graph.output[0].name
    sim_path = os.path.join(output_dir, "whitebox_sim.onnx")
    try:
        model_simp, check = simplify(model)
        onnx.save(model_simp if check else model, sim_path)
        print(f"✅ ONNX simplified: {os.path.getsize(sim_path)/1024/1024:.1f} MB")
    except Exception as e:
        print(f"⚠️ Simplify skipped: {e}")
        shutil.copy(raw_onnx, sim_path)

    # 4. ONNX → NCNN via PNNX
    pnnx_cmd = ["pnnx", sim_path, "inputshape=[1,512,512,3]", "device=cpu", "fp16=0", "optlevel=2"]
    if not run_with_log(pnnx_cmd, 600, "PNNX conversion"):
        sys.exit(1)
    
    pnnx_param = sim_path.replace(".onnx", ".ncnn.param")
    pnnx_bin = sim_path.replace(".onnx", ".ncnn.bin")
    if not os.path.exists(pnnx_param) or os.path.getsize(pnnx_bin) < 1024*1024:
        print("❌ PNNX output invalid"); sys.exit(1)

    # 5. Safe blob name replace
    param_path = os.path.join(output_dir, "whitebox.param")
    bin_path = os.path.join(output_dir, "whitebox.bin")
    shutil.copy(pnnx_param, param_path)
    shutil.copy(pnnx_bin, bin_path)
    
    with open(param_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith("#") or not line.strip():
            new_lines.append(line); continue
        parts = line.split()
        if len(parts) < 5: new_lines.append(line); continue
        try: in_count, out_count = int(parts[2]), int(parts[3])
        except: new_lines.append(line); continue
        idx = 4; new_parts = parts[:idx]
        for i in range(in_count):
            blob = parts[idx + i]
            new_parts.append("in0" if blob == onnx_input_name else blob)
        for i in range(out_count):
            blob = parts[idx + in_count + i]
            new_parts.append("out0" if blob == onnx_output_name else blob)
        new_lines.append(" ".join(new_parts) + "\n")
    with open(param_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # 6. Log final blobs
    ncnn_in = ncnn_out = None
    for line in lines:
        if line.startswith("Input"): ncnn_in = line.split()[-1]
        ncnn_out = line.split()[-1]
    if ncnn_in and ncnn_out:
        print(f"\n🔑 NCNN Input Blob: '{ncnn_in}'")
        print(f"🔑 NCNN Output Blob: '{ncnn_out}'")
        print(f"   Update C++: ex.input(\"{ncnn_in}\", in) / ex.extract(\"{ncnn_out}\", out)")

    print(f"\n✅ Output: whitebox.param ({os.path.getsize(param_path)/1024:.1f} KB), whitebox.bin ({os.path.getsize(bin_path)/1024/1024:.1f} MB)")

if __name__ == "__main__":
    main()
