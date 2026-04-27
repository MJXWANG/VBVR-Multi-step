#!/bin/bash
# =============================================================================
# Run A: Continue training on OTHER HALF of data (samples 5001-10000 per task)
# =============================================================================
# Continues from Run 1 LoRA checkpoint (step-21250) on the remaining 170K
# samples. After this, the model has seen all 340K samples.
#
# Prerequisites:
#   1. Run 1 completed (outputs/multistep/high_noise/step-21250.safetensors)
#   2. Generate annotations: --skip_samples 5000 --max_samples 5000
#   3. Kill vLLM before running
# =============================================================================

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_CONFIG_PATH="${REPO_DIR}/configs/multistep_dataset.json"

# Training parameters
HEIGHT=384
WIDTH=384
NUM_FRAMES=33
DATASET_REPEAT=1
LEARNING_RATE=1e-5
NUM_EPOCHS=1
LORA_RANK=32
SAVE_STEPS=500

# Output directories
# Output directories (separate from Run 1)
HIGH_NOISE_OUTPUT_PATH="${REPO_DIR}/outputs/multistep_runA/high_noise"
LOW_NOISE_OUTPUT_PATH="${REPO_DIR}/outputs/multistep_runA/low_noise"

# LoRA checkpoints from Run 1 (continue training)
LORA_CKPT_HIGH="${REPO_DIR}/outputs/multistep/high_noise/step-21250.safetensors"
LORA_CKPT_LOW="${REPO_DIR}/outputs/multistep/low_noise/step-21250.safetensors"

# Distributed training
NUM_GPUS=${NUM_GPUS:-8}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}
NUM_PROCESSES=$((NUM_GPUS * NUM_NODES))

REMOVE_PREFIX_IN_CKPT="pipe.dit."

# Verify prerequisites
if [ ! -f "${DATASET_CONFIG_PATH}" ]; then
    echo "ERROR: ${DATASET_CONFIG_PATH} not found. Run generate_multistep_annotations.py first."
    exit 1
fi

# Use locally downloaded model — no network needed
export DIFFSYNTH_MODEL_BASE_PATH="${REPO_DIR}/models"
export DIFFSYNTH_SKIP_DOWNLOAD="True"
export DIFFSYNTH_DOWNLOAD_SOURCE="huggingface"

echo "=================================="
echo "VBVR-Multi-step Training"
echo "=================================="
echo "Model:           Wan-AI/Wan2.2-I2V-A14B (local: ${DIFFSYNTH_MODEL_BASE_PATH})"
echo "Dataset config:  ${DATASET_CONFIG_PATH}"
echo "Resolution:      ${WIDTH}x${HEIGHT}, ${NUM_FRAMES} frames"
echo "NUM_PROCESSES:   ${NUM_PROCESSES} (${NUM_GPUS} GPUs x ${NUM_NODES} nodes)"
echo "Learning Rate:   ${LEARNING_RATE}"
echo "LoRA Rank:       ${LORA_RANK}"
echo "Save every:      ${SAVE_STEPS} steps"
echo "High Noise out:  ${HIGH_NOISE_OUTPUT_PATH}"
echo "Low Noise out:   ${LOW_NOISE_OUTPUT_PATH}"
echo "=================================="

# ---- Train High Noise Model LoRA ----
echo "[Step 1/2] Training High Noise Model LoRA (timestep: 0 ~ 0.358)"

cd ${REPO_DIR}
mkdir -p ${HIGH_NOISE_OUTPUT_PATH}
accelerate launch \
    --multi_gpu \
    --num_processes ${NUM_PROCESSES} \
    --num_machines ${NUM_NODES} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --machine_rank ${NODE_RANK} \
    ${REPO_DIR}/examples/wanvideo/model_training/train.py \
    --dataset_config_path ${DATASET_CONFIG_PATH} \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --dataset_repeat ${DATASET_REPEAT} \
    --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --remove_prefix_in_ckpt ${REMOVE_PREFIX_IN_CKPT} \
    --output_path ${HIGH_NOISE_OUTPUT_PATH} \
    --lora_base_model 'dit' \
    --lora_target_modules 'q,k,v,o,ffn.0,ffn.2' \
    --lora_rank ${LORA_RANK} \
    --lora_checkpoint ${LORA_CKPT_HIGH} \
    --extra_inputs 'input_image' \
    --max_timestep_boundary 0.358 \
    --min_timestep_boundary 0 \
    --data_file_keys 'clip_path' \
    --save_steps ${SAVE_STEPS}

echo "[Step 1/2] High Noise Model training complete."

# ---- Train Low Noise Model LoRA ----
echo "[Step 2/2] Training Low Noise Model LoRA (timestep: 0.358 ~ 1.0)"

mkdir -p ${LOW_NOISE_OUTPUT_PATH}
accelerate launch \
    --multi_gpu \
    --num_processes ${NUM_PROCESSES} \
    --num_machines ${NUM_NODES} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --machine_rank ${NODE_RANK} \
    ${REPO_DIR}/examples/wanvideo/model_training/train.py \
    --dataset_config_path ${DATASET_CONFIG_PATH} \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --dataset_repeat ${DATASET_REPEAT} \
    --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --remove_prefix_in_ckpt ${REMOVE_PREFIX_IN_CKPT} \
    --output_path ${LOW_NOISE_OUTPUT_PATH} \
    --lora_base_model 'dit' \
    --lora_target_modules 'q,k,v,o,ffn.0,ffn.2' \
    --lora_rank ${LORA_RANK} \
    --lora_checkpoint ${LORA_CKPT_LOW} \
    --extra_inputs 'input_image' \
    --max_timestep_boundary 1 \
    --min_timestep_boundary 0.358 \
    --data_file_keys 'clip_path' \
    --save_steps ${SAVE_STEPS}

echo "[Step 2/2] Low Noise Model training complete."
echo "All training done!"
