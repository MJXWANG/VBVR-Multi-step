# VBVR-Multi-step: Multi-step Video Reasoning

Train video generation models to perform multi-step reasoning — solving mazes, executing logic puzzles, simulating physics, and constructing geometric proofs — all as video generation.

Built on [VBVR-Wan2.2](https://github.com/Video-Reason/VBVR-Wan2.2) (DiffSynth framework). Fine-tunes **Wan2.2-I2V-A14B** (14B parameter dual-DiT video diffusion model) with LoRA adapters on 170K multi-step reasoning videos.

---

## 📦 Code & Data Release for NeurIPS 2026 ED Submission

This repository hosts the **training recipe** for VBVR-MultiStep. The full code and data release spans the following components:

| Component | Location |
|---|---|
| **Training code** (this repo) | https://github.com/MJXWANG/VBVR-Multi-step |
| **Inference harness** (`VBVR-InferKit`) | https://github.com/MJXWANG/VBVR-InferKit |
| **36 task data generators** | https://github.com/vbvr-datafactory (`Multi-01_*` … `Multi-36_*`) |
| **Training corpus + 180-instance evaluation split** | https://huggingface.co/datasets/Video-Reason/VBVR-MultiStep |
| **Reasoning-tuned base model** (`VBVR-Wan2.2`, prior-suite checkpoint) | https://github.com/Video-Reason/VBVR-Wan2.2 |

The training recipe and per-experiment hyperparameters are documented in §6 and Appendix H of the paper.

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/MJXWANG/VBVR-Multi-step.git
cd VBVR-Multi-step
pip install -e .
```

### 2. Download Base Model

```bash
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./models/Wan-AI/Wan2.2-I2V-A14B
```

This is ~73 GB. The model includes high-noise DiT, low-noise DiT, T5 text encoder, and VAE.

### 3. Prepare Training Data

The training data should follow this structure:

```
data/multistep/
├── Multi-01_maze_shortest_path_data-generator/
│   └── Multi-01_maze_shortest_path_data-generator_task/
│       ├── Multi-01_..._00000000/
│       │   ├── first_frame.png       (1024×1024)
│       │   ├── ground_truth.mp4      (1024×1024, ~34 frames, 16fps)
│       │   ├── final_frame.png
│       │   ├── prompt.txt            ([Scenario] + [Rules] + [Task])
│       │   └── question_metadata.json
│       ├── Multi-01_..._00000001/
│       └── ...
├── Multi-02_maze_circular_route_data-generator/
└── ... (34 task types)
```

### 4. Generate Annotation Configs

```bash
python scripts/generate_multistep_annotations.py \
    --data_root data/multistep \
    --output_configs configs/per_tasks \
    --output_dataset_config configs/multistep_dataset.json \
    --max_samples 5000
```

This creates per-task JSON annotation files and a dataset config. Use `--max_samples` to limit samples per task (we used 5,000 per task = 170K total).

### 5. Verify Setup

```bash
python scripts/test_model_load.py
```

This checks: model files exist, JSON configs parse correctly, dataset annotations are valid, video files are accessible. It also generates `configs/model_paths_high_noise.json` and `configs/model_paths_low_noise.json`.

### 6. Train

```bash
# Kill any existing GPU-occupying processes first
# Then run:
bash scripts/wan2.2-I2V-14B_multistep.sh
```

Training runs in two phases (Wan2.2-I2V-A14B uses a dual-DiT architecture):

| Phase | Model | Timestep Range | Description |
|-------|-------|---------------|-------------|
| Step 1/2 | High Noise (`dit`) | 0 – 0.358 | Learns rough structure/layout |
| Step 2/2 | Low Noise (`dit2`) | 0.358 – 1.0 | Learns fine details/sharpness |

**Key environment variable:** The training script sets `DIFFSYNTH_MODEL_BASE_PATH` to load models from the local `models/` directory. No network access is needed during training.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Wan-AI/Wan2.2-I2V-A14B (14B params) |
| Resolution | 384 × 384 |
| Frames | 33 (matching source video length) |
| LoRA rank | 32 |
| LoRA targets | q, k, v, o, ffn.0, ffn.2 |
| Learning rate | 1e-5 |
| Batch size | 1 per GPU |
| Epochs | 1 |
| Checkpoint interval | Every 500 steps |

### Hardware Requirements

Tested on 8× NVIDIA H200 (141 GB HBM3e each). Total training time: ~60 hours (30h per phase).

Minimum: 8 GPUs with ≥40 GB VRAM each (with gradient checkpointing enabled, which is forced on automatically).

### Outputs

After training, LoRA checkpoints are saved to:

```
outputs/multistep/
├── high_noise/
│   ├── step-500.safetensors
│   ├── step-1000.safetensors
│   ├── ...
│   └── epoch-0.safetensors
└── low_noise/
    ├── step-500.safetensors
    ├── ...
    └── epoch-0.safetensors
```

### Inference

```bash
python examples/wanvideo/model_training/validate_lora/eval_vbvr_bench.py \
    --eval_root ./data/VBVR-Bench \
    --output_root ./outputs/eval/VBVR-Multi-step \
    --high_noise_lora_path ./outputs/multistep/high_noise/epoch-0.safetensors \
    --low_noise_lora_path ./outputs/multistep/low_noise/epoch-0.safetensors
```

## Task Types (34)

| Category | Tasks |
|----------|-------|
| **Planning / Pathfinding** | Maze shortest path, circular route, marker drawing, dual-dot pathfinding, route tracing, BFS tree traversal, Sokoban, sliding puzzle, Tower of Hanoi, snake routing, TSP reward collection |
| **Logic / Computation** | Ordinal number sequence, word search, Sudoku, Numbrix, orthogonal Latin square, tents & trees, Turing machine execution, Langton's ant, chained math, pointer chasing, code pipeline, Conway's Game of Life |
| **Physics** | Communicating vessels, multiple bounces, elastic bouncing, elastic collision, block sliding friction, target after reflection |
| **Geometry** | Light reflection ray tracing, line intersection, perpendicular bisector, triangle orthocenter/incenter/circumcenter construction |

## Offline Training (Air-Gapped Environments)

If your GPU server has no internet access:

1. Download the base model on a machine with internet:
   ```bash
   huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./models/Wan-AI/Wan2.2-I2V-A14B
   ```

2. Transfer the `models/` directory to the GPU server (shared filesystem, rsync, etc.)

3. The training script automatically sets `DIFFSYNTH_MODEL_BASE_PATH` and `DIFFSYNTH_SKIP_DOWNLOAD=True` — no network calls during training.

**Note:** The code sets `redirect_common_files=False` in the model loading pipeline. This prevents DiffSynth from redirecting T5/VAE files to a different HuggingFace repository, which would fail in offline environments.

## Acknowledgements

Built on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and [VBVR-Wan2.2](https://github.com/Video-Reason/VBVR-Wan2.2).
