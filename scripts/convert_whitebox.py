# scripts/convert_whitebox.py (thay đổi phần export_script)

    export_script = '''
import os
import sys
import tensorflow as tf

# Force TF1 behavior
tf1 = tf.compat.v1
tf1.disable_eager_execution()

# Patch tf.contrib.slim → tf_slim
import tf_slim as slim
tf.contrib = type(sys)('contrib')
tf.contrib.slim = slim

# Add test_code to path
test_code_dir = "{test_code_dir}"
sys.path.insert(0, test_code_dir)

# Patch network.py imports
import types
import importlib.util

# Load network module manually to handle tf.contrib.slim
spec = importlib.util.spec_from_file_location(
    "network", os.path.join(test_code_dir, "network.py"))
network_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(network_mod)

unet_generator = network_mod.unet_generator

print("Building graph...")

# Build graph — White-Box uses 512x512 input
input_ph = tf1.placeholder(tf.float32, [1, 512, 512, 3], name="input")

# Build generator
output = unet_generator(input_ph, channel=32, num_blocks=4, name="generator", reuse=False)

# Output
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

# Print tensor names
for node in graph_def.node:
    if node.op == "Placeholder":
        print(f"Input: {{node.name}}")
    if "output" in node.name.lower():
        print(f"Output candidate: {{node.name}} (op={{node.op}})")

sess.close()
print("Export DONE")
'''.format(
        test_code_dir=test_code_dir,
        ckpt_path=ckpt_path,
        output_dir=output_dir,
    )
