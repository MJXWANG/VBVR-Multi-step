# VBVR-Multi-step: Multi-step Video Reasoning

Train video generation models to perform multi-step reasoning — solving mazes, executing logic puzzles, simulating physics, and constructing geometric proofs — all as video generation.

Built on [VBVR-Wan2.2](https://github.com/Video-Reason/VBVR-Wan2.2) (DiffSynth framework). Fine-tunes **Wan2.2-I2V-A14B** (14B parameter dual-DiT video diffusion model) with LoRA adapters on 170K multi-step reasoning videos.

---

## 📦 Code & Data Release — VBVR-MultiStep

This repository hosts the **training recipe** for VBVR-MultiStep. The full code and data release spans the following components:

### Core artifacts

| Component | Location |
|---|---|
| **Training code** (this repo) | https://github.com/MJXWANG/VBVR-Multi-step |
| **Inference harness** (`VBVR-InferKit`) | https://github.com/MJXWANG/VBVR-InferKit |
| **Training corpus** (~360k samples) | https://huggingface.co/datasets/Video-Reason/VBVR-MultiStep |
| **Frozen 180-instance evaluation split** | https://huggingface.co/datasets/Video-Reason/VBVR-MultiStep-Bench |
| **36 task data generators** | See [task generator index](#36-task-generators) below |
| **Reasoning-tuned base model** (`VBVR-Wan2.2`, prior-suite checkpoint) | https://github.com/Video-Reason/VBVR-Wan2.2 |

The training recipe and per-experiment hyperparameters are documented in §6 and Appendix H of the paper.

### Trained checkpoints
The 172-checkpoint inventory (Exp\,2 + Exp\,3 × {high, low}-noise expert × 43 saves; ~54 GB of `.safetensors` files) is archived internally. A representative subset will be released upon acceptance.

### 36 Task Generators

Each task is released as a standalone repository under [`vbvr-datafactory`](https://github.com/vbvr-datafactory) with its own README, generator core, and example samples.

| ID | Slug | Family | Repository |
|---|---|---|---|
| Multi-01 | `maze_shortest_path` | Navigation | https://github.com/vbvr-datafactory/Multi-01_maze_shortest_path_data-generator |
| Multi-02 | `maze_circular_route` | Navigation | https://github.com/vbvr-datafactory/Multi-02_maze_circular_route_data-generator |
| Multi-03 | `maze_marker_drawing` | Navigation | https://github.com/vbvr-datafactory/Multi-03_maze_marker_drawing_data-generator |
| Multi-04 | `dual_dot_pathfinding` | Navigation | https://github.com/vbvr-datafactory/Multi-04_dual_dot_pathfinding_data-generator |
| Multi-05 | `maze_route_tracing` | Navigation | https://github.com/vbvr-datafactory/Multi-05_maze_route_tracing_data-generator |
| Multi-06 | `bfs_tree_traversal` | Navigation | https://github.com/vbvr-datafactory/Multi-06_bfs_tree_traversal_data-generator |
| Multi-07 | `sokoban_planning` | Planning | https://github.com/vbvr-datafactory/Multi-07_sokoban_planning_data-generator |
| Multi-08 | `sliding_puzzle` | Planning | https://github.com/vbvr-datafactory/Multi-08_sliding_puzzle_data-generator |
| Multi-09 | `tower_of_hanoi` | Planning | https://github.com/vbvr-datafactory/Multi-09_tower_of_hanoi_data-generator |
| Multi-10 | `snake_dynamic_routing` | Planning | https://github.com/vbvr-datafactory/Multi-10_snake_dynamic_routing_data-generator |
| Multi-11 | `tsp_reward_collection` | Planning | https://github.com/vbvr-datafactory/Multi-11_tsp_reward_collection_data-generator |
| Multi-12 | `ordinal_number_sequence` | Planning | https://github.com/vbvr-datafactory/Multi-12_ordinal_number_sequence_data-generator |
| Multi-13 | `wordsearch_path` | CSP | https://github.com/vbvr-datafactory/Multi-13_wordsearch_path_data-generator |
| Multi-14 | `sudoku_logic` | CSP | https://github.com/vbvr-datafactory/Multi-14_sudoku_logic_data-generator |
| Multi-15 | `numbrix_pathfilling` | CSP | https://github.com/vbvr-datafactory/Multi-15_numbrix_pathfilling_data-generator |
| Multi-16 | `orthogonal_latin_square` | CSP | https://github.com/vbvr-datafactory/Multi-16_orthogonal_latin_square_data-generator |
| Multi-17 | `hashi_bridges` | CSP | https://github.com/vbvr-datafactory/Multi-17_hashi_bridges_data-generator |
| Multi-18 | `tents_and_trees` | CSP | https://github.com/vbvr-datafactory/Multi-18_tents_and_trees_data-generator |
| Multi-19 | `turing_machine_execution` | Execution | https://github.com/vbvr-datafactory/Multi-19_turing_machine_execution_data-generator |
| Multi-20 | `langtons_ant_simulation` | Execution | https://github.com/vbvr-datafactory/Multi-20_langtons_ant_simulation_data-generator |
| Multi-21 | `chained_math_calculation` | Execution | https://github.com/vbvr-datafactory/Multi-21_chained_math_calculation_data-generator |
| Multi-22 | `pointer_chasing_arrows` | Execution | https://github.com/vbvr-datafactory/Multi-22_pointer_chasing_arrows_data-generator |
| Multi-23 | `chained_code_pipeline` | Execution | https://github.com/vbvr-datafactory/Multi-23_chained_code_pipeline_data-generator |
| Multi-24 | `conways_game_of_life` | Execution | https://github.com/vbvr-datafactory/Multi-24_conways_game_of_life_data-generator |
| Multi-25 | `light_reflection_ray_tracing` | Geometry | https://github.com/vbvr-datafactory/Multi-25_light_reflection_ray_tracing_data-generator |
| Multi-26 | `line_intersection_construction` | Geometry | https://github.com/vbvr-datafactory/Multi-26_line_intersection_construction_data-generator |
| Multi-27 | `perpendicular_bisector_construction` | Geometry | https://github.com/vbvr-datafactory/Multi-27_perpendicular_bisector_construction_data-generator |
| Multi-28 | `triangle_orthocenter_construction` | Geometry | https://github.com/vbvr-datafactory/Multi-28_triangle_orthocenter_construction_data-generator |
| Multi-29 | `triangle_incenter_construction` | Geometry | https://github.com/vbvr-datafactory/Multi-29_triangle_incenter_construction_data-generator |
| Multi-30 | `triangle_circumcenter_construction` | Geometry | https://github.com/vbvr-datafactory/Multi-30_triangle_circumcenter_construction_data-generator |
| Multi-31 | `fluid_communicating_vessels` | Physics | https://github.com/vbvr-datafactory/Multi-31_fluid_communicating_vessels_data-generator |
| Multi-32 | `multiple_bounces_target` | Physics | https://github.com/vbvr-datafactory/Multi-32_multiple_bounces_target_data-generator |
| Multi-33 | `elastic_bouncing_trajectory` | Physics | https://github.com/vbvr-datafactory/Multi-33_elastic_bouncing_trajectory_data-generator |
| Multi-34 | `elastic_collision_kinematics` | Physics | https://github.com/vbvr-datafactory/Multi-34_elastic_collision_kinematics_data-generator |
| Multi-35 | `block_sliding_friction` | Physics | https://github.com/vbvr-datafactory/Multi-35_block_sliding_friction_data-generator |
| Multi-36 | `target_after_reflection` | Physics | https://github.com/vbvr-datafactory/Multi-36_target_after_reflection_data-generator |

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

## Reasoning Families (36 tasks across 6 families)

The 36 tasks (`Multi-01` … `Multi-36`) are organized into six reasoning families, matching §3 of the paper. See the [36 Task Generators table](#36-task-generators) above for per-task repositories.

| Family | Tasks (6 each) |
|---|---|
| **Navigation** | Maze shortest path, maze circular route, maze marker drawing, dual-dot pathfinding, maze route tracing, BFS tree traversal |
| **Planning** | Sokoban, sliding puzzle, Tower of Hanoi, snake dynamic routing, TSP reward collection, ordinal number sequence |
| **CSP** (constraint satisfaction) | Word search path, Sudoku logic, Numbrix path-filling, orthogonal Latin square, Hashi bridges, tents and trees |
| **Execution** | Turing machine execution, Langton's ant simulation, chained math calculation, pointer chasing arrows, chained code pipeline, Conway's Game of Life |
| **Geometry** | Light reflection ray tracing, line intersection construction, perpendicular bisector, triangle orthocenter, triangle incenter, triangle circumcenter |
| **Physics** | Fluid communicating vessels, multiple bounces target, elastic bouncing trajectory, elastic collision kinematics, block sliding friction, target after reflection |

Note: training in this repository covers **34 of the 36** tasks per the paper §6 recipe; all 36 generators remain part of the released benchmark suite.

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
