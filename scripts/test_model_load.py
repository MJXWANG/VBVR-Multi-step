"""Quick test: can we load the model and run one forward pass?

Run on GPU server:
    source /home/sankuai/conda/etc/profile.d/conda.sh
    conda activate .../envs/hippocampal
    export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
    cd .../VBVR-Multi-step
    python scripts/test_model_load.py
"""
import torch, os, sys, json

os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"

BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/VBVR-Multi-step"
MODEL_DIR = f"{BASE}/models/Wan-AI/Wan2.2-I2V-A14B"

print("=== Step 1: Check model files exist ===")
for subdir in ["high_noise_model", "low_noise_model"]:
    files = sorted(os.listdir(f"{MODEL_DIR}/{subdir}"))
    print(f"  {subdir}/: {len(files)} files — {files[:3]}...")
for f in ["models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"]:
    path = f"{MODEL_DIR}/{f}"
    exists = os.path.exists(path)
    size = os.path.getsize(path) / 1e9 if exists else 0
    print(f"  {f}: {'OK' if exists else 'MISSING'} ({size:.1f} GB)")

print("\n=== Step 2: Test --model_paths JSON format ===")
# Build the JSON list that --model_paths expects
import glob
high_noise_files = sorted(glob.glob(f"{MODEL_DIR}/high_noise_model/diffusion_pytorch_model*.safetensors"))
model_paths_list = high_noise_files + [
    f"{MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth",
    f"{MODEL_DIR}/Wan2.1_VAE.pth",
]
json_str = json.dumps(model_paths_list)
print(f"  JSON string (first 200 chars): {json_str[:200]}...")
# Verify it parses back
parsed = json.loads(json_str)
print(f"  Parsed: {len(parsed)} paths")
for p in parsed:
    print(f"    {os.path.basename(p)}: exists={os.path.exists(p)}")

print("\n=== Step 3: Test model import ===")
try:
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    print("  WanVideoPipeline imported OK")
except Exception as e:
    print(f"  IMPORT FAILED: {e}")
    sys.exit(1)

print("\n=== Step 4: Test dataset config ===")
config_path = f"{BASE}/configs/multistep_dataset.json"
if os.path.exists(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    print(f"  Dataset config: {len(cfg)} tasks")
    first_task = list(cfg.keys())[0]
    annot_path = cfg[first_task]["annotation"]
    if os.path.exists(annot_path):
        with open(annot_path) as f:
            annots = json.load(f)
        print(f"  First task ({first_task}): {len(annots)} samples")
        print(f"  Sample: {annots[0]}")
        clip_path = os.path.join(cfg[first_task]["root"], annots[0]["clip_path"])
        print(f"  Video path: {clip_path}")
        print(f"  Video exists: {os.path.exists(clip_path)}")
    else:
        print(f"  MISSING annotation: {annot_path}")
else:
    print(f"  MISSING config: {config_path}")

print("\n=== Step 5: Write model_paths JSON file ===")
json_path = f"{BASE}/configs/model_paths_high_noise.json"
with open(json_path, "w") as f:
    json.dump(model_paths_list, f)
print(f"  Written to {json_path}")

low_noise_files = sorted(glob.glob(f"{MODEL_DIR}/low_noise_model/diffusion_pytorch_model*.safetensors"))
low_paths_list = low_noise_files + [
    f"{MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth",
    f"{MODEL_DIR}/Wan2.1_VAE.pth",
]
json_path_low = f"{BASE}/configs/model_paths_low_noise.json"
with open(json_path_low, "w") as f:
    json.dump(low_paths_list, f)
print(f"  Written to {json_path_low}")

print("\n=== ALL CHECKS PASSED ===")
