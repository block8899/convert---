"""
White-Box Cartoonization TF1 -> Frozen PB -> ONNX -> NCNN
✅ Fixed: inputshape, safe name replace, blob logging, error handling
"""
import os, sys, subprocess, shutil, re, onnx
from onnxsim import simplify

def log(msg, level="INFO"):
    print(f"[{level}] {msg}")

def run_cmd(cmd, timeout=300, desc=""):
    """Helper: run subprocess with logging + error handling"""
    log(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if res.stdout:
            tail = res.stdout[-300:] if len(res.stdout) > 300 else res.stdout
            log(f"  stdout: {tail}")
        if res.returncode != 0:
            err_tail = res.stderr[-500:] if len(res.stderr) > 500 else res.stderr
            log(f"  ❌ {desc} failed:\n{err_tail}", "ERROR")
            return None
        return res
    except subprocess.TimeoutExpired:
        log(f"  ❌ {desc} timed out ({timeout}s)", "ERROR")
        return None

def safe_replace_blob_names(param_path, old_in, old_out, new_in="in0", new_out="out0"):
    """Safe replace input/output blob names in NCNN param file"""
    with open(param_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        # Skip comments and empty lines
        if line.strip().startswith("#") or not line.strip():
            new_lines.append(line)
            continue
        
        parts = line.split()
        if len(parts) < 5:  # Not a valid layer line
            new_lines.append(line)
            continue
        
        # parts: [LayerType, LayerName, InputCount, OutputCount, blobs...]
        layer_type, layer_name = parts[0], parts[1]
        try:
            in_count, out_count = int(parts[2]), int(parts[3])
        except ValueError:
            new_lines.append(line)
            continue
        
        blob_start_idx = 4
        new_parts = parts[:blob_start_idx]
        
        # Replace input blobs
        for i in range(in_count):
            blob = parts[blob_start_idx + i]
            if blob == old_in:
                new_parts.append(new_in)
            else:
                new_parts.append(blob)
        
        # Replace output blobs
        for i in range(out_count):
            blob = parts[blob_start_idx + in_count + i]
            if blob == old_out:
                new_parts.append(new_out)
            else:
                new_parts.append(blob)
        
        new_lines.append(" ".join(new_parts) + "\n")
    
    with open(param_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    log(f"  ✅ Replaced: '{old_in}'→'{new_in}', '{old_out}'→'{new_out}'")

def main():
    log("=== White-Box Cartoonization TF1 → NCNN ===")

    repo_dir = "repo_wbc"
    test_code_dir = os.path.join(repo_dir, "test_code")
    ckpt_path = os.path.join(test_code_dir, "saved_models", "model-33999")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(ckpt_path + ".index"):
        log(f"MISSING checkpoint: {ckpt_path}.index", "ERROR")
        sys.exit(1)

    log(f"Checkpoint: {ckpt_path}")

    # 1. Export Frozen PB
    log("1. Exporting Frozen PB...")
    frozen_pb = os.path.join(output_dir, "whitebox_frozen.pb")
    export_py = os.path.join(output_dir, "_export_frozen.py")

    export_code = f'''
import os, sys, tensorflow as tf
tf1 = tf.compat.v1
tf1.disable_eager_execution()

# Mock tf.contrib.slim with tf_slim
import tf_slim as slim
contrib_mock = type(sys)('tensorflow.contrib')
contrib_mock.slim = slim
sys.modules['tensorflow.contrib'] = contrib_mock
sys.modules['tensorflow.contrib.slim'] = slim

# Re-export TF1 APIs
for attr in ['variable_scope', 'get_variable', 'placeholder', 'Session', 'Saver', 'GraphDef']:
    setattr(tf, attr, getattr(tf1, attr))
setattr(tf.image, 'resize_bilinear', tf1.image.resize_bilinear)
setattr(tf.image, 'resize_nearest_neighbor', tf1.image.resize_nearest_neighbor)

sys.path.insert(0, r'{test_code_dir}')
import network

print("Building graph...")
input_ph = tf1.placeholder(tf.float32, [1, 512, 512, 3], name='input')
output = network.unet_generator(input_ph, channel=32, num_blocks=4, name='generator', reuse=False)
output = tf.identity(output, name='output')
print(f"Output shape: {{output.shape}}")

print("Loading checkpoint...")
saver = tf1.train.Saver()
sess = tf1.Session()
saver.restore(sess, r'{ckpt_path}')

print("Freezing graph...")
graph_def = tf1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])
with tf.io.gfile.GFile(r'{frozen_pb}', 'wb') as f:
    f.write(graph_def.SerializeToString())
print(f"Frozen PB: {{os.path.getsize(r'{frozen_pb}')/1024/1024:.1f}} MB")

# Print IO names for reference
for node in graph_def.node:
    if node.op == 'Placeholder': print(f"INPUT_NODE: {{node.name}}")
    if node.name == 'output': print(f"OUTPUT_NODE: {{node.name}}")
sess.close()
'''
    with open(export_py, "w") as f:
        f.write(export_code)

    if not run_cmd([sys.executable, export_py], timeout=300, desc="Export Frozen PB"):
        sys.exit(1)
    if not os.path.exists(frozen_pb):
        log("Frozen PB not created!", "ERROR"); sys.exit(1)

    # 2. Frozen PB -> ONNX
    log("2. Converting PB -> ONNX...")
    raw_onnx = os.path.join(output_dir, "whitebox_raw.onnx")
    
    output_candidates = ["output:0", "generator/output:0", "output_1:0"]
    success = False
    
    for out_name in output_candidates:
        cmd = [
            sys.executable, "-m", "tf2onnx.convert",
            "--graphdef", frozen_pb,
            "--output", raw_onnx,
            "--opset", "11",
            "--inputs", "input:0",
            "--outputs", out_name,
        ]
        log(f"  Trying output: {out_name}")
        if run_cmd(cmd, timeout=300, desc=f"tf2onnx ({out_name})"):
            if os.path.exists(raw_onnx) and os.path.getsize(raw_onnx) > 10*1024*1024:
                success = True
                log(f"  ✅ Success with: {out_name}")
                break
    
    if not success:
        log("All ONNX conversion attempts failed!", "ERROR"); sys.exit(1)

    # 3. Simplify ONNX
    log("3. Simplifying ONNX...")
    model = onnx.load(raw_onnx)
    onnx_input_name = model.graph.input[0].name
    onnx_output_name = model.graph.output[0].name
    log(f"  ONNX Input: '{onnx_input_name}', Output: '{onnx_output_name}'")
    
    sim_path = os.path.join(output_dir, "whitebox_sim.onnx")
    try:
        model_simp, check = simplify(model)
        if check:
            onnx.save(model_simp, sim_path)
            log(f"  Simplified: {os.path.getsize(sim_path)/1024/1024:.1f} MB")
        else:
            log("  Simplify check failed, using original", "WARN")
            shutil.copy(raw_onnx, sim_path)
    except Exception as e:
        log(f"  Simplify exception: {e}", "WARN")
        shutil.copy(raw_onnx, sim_path)

    # 4. ONNX -> NCNN via PNNX ✅ FIXED
    log("4. Converting ONNX -> NCNN via PNNX...")
    
    pnnx_cmd = [
        "pnnx", sim_path,
        "inputshape=[1,512,512,3]",  # ✅ Khớp frozen graph
        "device=cpu",                 # ✅ Tương thích Android
        "fp16=0",                     # ✅ Tránh numerical issues
        "optlevel=2"                 # ✅ Safe optimization
    ]
    
    if not run_cmd(pnnx_cmd, timeout=600, desc="PNNX"):  # 10 phút cho PNNX
        sys.exit(1)
    
    pnnx_param = sim_path.replace(".onnx", ".ncnn.param")
    pnnx_bin = sim_path.replace(".onnx", ".ncnn.bin")
    
    if not os.path.exists(pnnx_param) or not os.path.getsize(pnnx_bin) > 1024*1024:
        log("PNNX output invalid!", "ERROR"); sys.exit(1)

    # 5. Copy + Fix names ✅ SAFE REPLACE
    param_path = os.path.join(output_dir, "whitebox.param")
    bin_path = os.path.join(output_dir, "whitebox.bin")
    shutil.copy(pnnx_param, param_path)
    shutil.copy(pnnx_bin, bin_path)
    
    safe_replace_blob_names(param_path, onnx_input_name, onnx_output_name)

    # 6. Extract & log NCNN blob names ✅
    log("5. Detecting NCNN blob names...")
    ncnn_in = ncnn_out = None
    with open(param_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    
    for line in lines:
        if line.startswith("Input"):
            ncnn_in = line.split()[-1]
        ncnn_out = line.split()[-1]  # Last layer output
    
    if ncnn_in and ncnn_out:
        log(f"  ✅ NCNN Input Blob: '{ncnn_in}'")
        log(f"  ✅ NCNN Output Blob: '{ncnn_out}'")
        log(f"  👉 Update C++: ex.input(\"{ncnn_in}\", in) / ex.extract(\"{ncnn_out}\", out)")
    else:
        log("  ⚠️ Blob detect failed. Open .param in Netron.", "WARN")

    # 7. Final summary
    log("\n=== Output ===")
    log(f"  whitebox.param: {os.path.getsize(param_path)/1024:.1f} KB")
    log(f"  whitebox.bin:   {os.path.getsize(bin_path)/1024/1024:.1f} MB")
    log(f"  Param lines: {len(lines)}")
    log("\n✅ White-Box Cartoonization conversion DONE!")

if __name__ == "__main__":
    main()
