"""White-Box Cartoonization: TF1 -> Frozen PB -> ONNX -> NCNN (Fixed)"""
import os, sys, subprocess, shutil, onnx

def run_cmd(cmd, timeout, desc):
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        print(f"❌ {desc} failed:\n{res.stderr[-800:]}")
        return False
    return True

def safe_replace_blob_names(param_path, old_in, old_out, new_in="in0", new_out="out0"):
    with open(param_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith("#") or not line.strip():
            new_lines.append(line); continue
        parts = line.split()
        if len(parts) < 5: new_lines.append(line); continue
        try: in_c, out_c = int(parts[2]), int(parts[3])
        except: new_lines.append(line); continue
        idx = 4
        new_parts = parts[:idx]
        for i in range(in_c):
            new_parts.append(new_in if parts[idx+i] == old_in else parts[idx+i])
        for i in range(out_c):
            new_parts.append(new_out if parts[idx+in_c+i] == old_out else parts[idx+in_c+i])
        new_lines.append(" ".join(new_parts) + "\n")
    with open(param_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"✅ Renamed blobs: '{old_in}'→'{new_in}', '{old_out}'→'{new_out}'")

def main():
    repo_dir = "repo_wbc"
    test_code_dir = os.path.join(repo_dir, "test_code")
    ckpt_path = os.path.join(test_code_dir, "saved_models", "model-33999")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(ckpt_path + ".index"):
        print(f"❌ Missing checkpoint: {ckpt_path}.index"); sys.exit(1)

    # 1. Export Frozen PB
    frozen_pb = os.path.join(output_dir, "whitebox_frozen.pb")
    export_py = os.path.join(output_dir, "_export_frozen.py")
    export_code = f'''
import os, sys, tensorflow as tf
tf1 = tf.compat.v1; tf1.disable_eager_execution()
import tf_slim as slim
tf.contrib = type(sys)("tensorflow.contrib"); tf.contrib.slim = slim
sys.modules["tensorflow.contrib"] = tf.contrib; sys.modules["tensorflow.contrib.slim"] = slim
for attr in ["variable_scope","get_variable","placeholder","Session","GraphDef"]: setattr(tf, attr, getattr(tf1, attr))
tf.train.Saver = tf1.train.Saver
tf.image.resize_bilinear = tf1.image.resize_bilinear
tf.image.resize_nearest_neighbor = tf1.image.resize_nearest_neighbor
sys.path.insert(0, r"{test_code_dir}")
import network
input_ph = tf1.placeholder(tf.float32, [1, 512, 512, 3], name="input")
output = network.unet_generator(input_ph, channel=32, num_blocks=4, name="generator", reuse=False)
output = tf.identity(output, name="output")
saver = tf1.train.Saver(); sess = tf1.Session()
saver.restore(sess, r"{ckpt_path}")
graph_def = tf1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
with tf.io.gfile.GFile(r"{frozen_pb}", "wb") as f: f.write(graph_def.SerializeToString())
print(f"✅ Frozen PB: {{os.path.getsize(r'{frozen_pb}')/1024/1024:.1f}} MB")
sess.close()
'''
    with open(export_py, "w") as f: f.write(export_code)
    if not run_cmd([sys.executable, export_py], 300, "Export Frozen PB") or not os.path.exists(frozen_pb):
        sys.exit(1)

    # 2. PB -> ONNX
    raw_onnx = os.path.join(output_dir, "whitebox_raw.onnx")
    success = False
    for out_name in ["output:0", "generator/output:0", "Identity:0"]:
        cmd = [sys.executable, "-m", "tf2onnx.convert", "--graphdef", frozen_pb, "--output", raw_onnx, "--opset", "11", "--inputs", "input:0", "--outputs", out_name]
        if run_cmd(cmd, 300, f"tf2onnx ({out_name})") and os.path.getsize(raw_onnx) > 5_000_000:
            success = True; break
    if not success: print("❌ ONNX conversion failed"); sys.exit(1)

    # 3. Skip onnxsim (giữ nguyên graph UNet) + Lấy tên blob
    sim_path = os.path.join(output_dir, "whitebox_sim.onnx")
    shutil.copy(raw_onnx, sim_path)
    print("✅ Skipped onnxsim to preserve UNet structure.")
    
    model = onnx.load(raw_onnx)
    onnx_in = model.graph.input[0].name
    onnx_out = model.graph.output[0].name

    # 4. ONNX -> NCNN
    pnnx_cmd = ["pnnx", sim_path, "inputshape=[1,512,512,3]", "device=cpu", "fp16=0", "optlevel=2"]
    if not run_cmd(pnnx_cmd, 600, "PNNX"): sys.exit(1)

    pnnx_param = sim_path.replace(".onnx", ".ncnn.param")
    pnnx_bin = sim_path.replace(".onnx", ".ncnn.bin")
    if not os.path.exists(pnnx_param) or os.path.getsize(pnnx_bin) < 1_000_000:
        print("❌ PNNX output invalid"); sys.exit(1)

    # 5. Safe Rename & Copy
    param_path = os.path.join(output_dir, "whitebox.param")
    bin_path = os.path.join(output_dir, "whitebox.bin")
    shutil.copy(pnnx_param, param_path); shutil.copy(pnnx_bin, bin_path)
    safe_replace_blob_names(param_path, onnx_in, onnx_out)

    # 6. Final Log
    ncnn_in = ncnn_out = None
    with open(param_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for l in lines:
        if l.startswith("Input"): ncnn_in = l.split()[-1]
        ncnn_out = l.split()[-1]
    if ncnn_in and ncnn_out:
        print(f"\n🔑 NCNN Input: '{ncnn_in}' | Output: '{ncnn_out}'")
        print(f"   C++: ex.input(\"{ncnn_in}\", in) / ex.extract(\"{ncnn_out}\", out)")
    print(f"\n✅ Done: param ({os.path.getsize(param_path)/1024:.1f}KB) | bin ({os.path.getsize(bin_path)/1024/1024:.1f}MB)")

if __name__ == "__main__":
    main()
