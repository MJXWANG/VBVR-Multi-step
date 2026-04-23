#!/usr/bin/env python3
"""Generate per-task annotation JSONs and dataset config for multi-step data.

Usage:
    python scripts/generate_multistep_annotations.py \
        --data_root data/multistep \
        --output_configs configs/per_tasks \
        --output_dataset_config configs/multistep_dataset.json
"""
import os, json, argparse
from pathlib import Path


def find_samples(task_dir: Path):
    """Find all samples in a task directory and return annotation entries."""
    entries = []
    task_subdir = task_dir / f"{task_dir.name}_task"
    if not task_subdir.exists():
        task_subdir = task_dir
    for sample_dir in sorted(task_subdir.iterdir()):
        if not sample_dir.is_dir():
            continue
        gt = sample_dir / "ground_truth.mp4"
        prompt_file = sample_dir / "prompt.txt"
        if not gt.exists() or not prompt_file.exists():
            continue
        rel_clip = str(gt.relative_to(task_dir.parent))
        text = prompt_file.read_text(encoding="utf-8").strip()
        entries.append({"clip_path": rel_clip, "text_annot": text})
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Root dir containing Multi-XX_* task folders")
    parser.add_argument("--output_configs", default="configs/per_tasks", help="Dir for per-task JSONs")
    parser.add_argument("--output_dataset_config", default="configs/multistep_dataset.json", help="Output dataset config")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_configs)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = {}
    total_samples = 0

    for task_dir in sorted(data_root.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("Multi-"):
            continue
        task_name = task_dir.name
        entries = find_samples(task_dir)
        if not entries:
            print(f"  SKIP {task_name}: no valid samples")
            continue

        annotation_path = out_dir / f"{task_name}.json"
        with open(annotation_path, "w") as f:
            json.dump(entries, f, indent=2)

        dataset_config[task_name] = {
            "root": str(data_root),
            "annotation": str(annotation_path),
        }
        total_samples += len(entries)
        print(f"  {task_name}: {len(entries)} samples")

    with open(args.output_dataset_config, "w") as f:
        json.dump(dataset_config, f, indent=4)

    print(f"\nTotal: {len(dataset_config)} tasks, {total_samples} samples")
    print(f"Dataset config: {args.output_dataset_config}")


if __name__ == "__main__":
    main()
