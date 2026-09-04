"""Measure Mask R-CNN deployment statistics for ClapperText runs."""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmocr.apis import MMOCRInferencer


def _extract_input_hw(cfg) -> tuple:
    pipeline = None
    if hasattr(cfg, "test_pipeline"):
        pipeline = cfg.test_pipeline
    elif hasattr(cfg, "test_dataloader") and "dataset" in cfg.test_dataloader:
        pipeline = cfg.test_dataloader["dataset"].get("pipeline")

    if not pipeline:
        return 640, 640

    for step in pipeline:
        if not isinstance(step, dict):
            continue
        if step.get("type") == "Resize" and "scale" in step:
            scale = step["scale"]
            if isinstance(scale, tuple) and len(scale) == 2:
                return scale[1], scale[0]
        if step.get("type") == "ShortScaleAspectJitter" and "short_size" in step:
            short_size = step["short_size"]
            return short_size, short_size
    return 640, 640


def _get_model(inferencer):
    textdet = getattr(inferencer, "textdet_inferencer", None)
    if textdet is not None and hasattr(textdet, "model"):
        return textdet.model
    for attribute in ("det_model", "model"):
        model = getattr(inferencer, attribute, None)
        if model is not None:
            return model
    return None


def get_deploy_stats(config_path, weights, image_dir, device="cuda"):
    cfg = Config.fromfile(config_path)
    raw_height, raw_width = map(int, _extract_input_hw(cfg))
    height = int(np.ceil(raw_height / 32) * 32)
    width = int(np.ceil(raw_width / 32) * 32)
    stats = {
        "params": None,
        "fps": None,
        "latency_ms": None,
        "mem_mb": None,
        "input_h": raw_height,
        "input_w": raw_width,
        "input_h_pad32": height,
        "input_w_pad32": width,
    }

    inferencer = MMOCRInferencer(det=config_path, det_weights=weights, device=device)
    model = _get_model(inferencer)
    if model is None:
        warnings.warn("Cannot extract the detection model from MMOCRInferencer")
    else:
        stats["params"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)

    extensions = {".jpg", ".jpeg", ".png"}
    images = sorted(
        str(path)
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )[:3]
    if not images:
        warnings.warn(f"No sample images found under {image_dir}")
        return stats

    use_cuda = device.startswith("cuda")
    with torch.no_grad():
        for _ in range(3):
            inferencer(images)
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

        repeats = 10
        start = time.perf_counter()
        for _ in range(repeats):
            inferencer(images)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    sample_count = repeats * len(images)
    stats["fps"] = round(sample_count / elapsed, 2)
    stats["latency_ms"] = round(1000 * elapsed / sample_count, 2)
    if use_cuda:
        stats["mem_mb"] = round(torch.cuda.max_memory_allocated(device) / 1e6, 1)
    return stats


def run_experiment(exp_root: Path, mode: str, image_dir: Path, device: str):
    config_path = next(iter(sorted(exp_root.glob("*.py"))), None)
    if config_path is None:
        raise FileNotFoundError(f"No config file found in {exp_root}")

    if mode == "zeroshot":
        weights = Config.fromfile(config_path).get("load_from")
        if not weights:
            raise ValueError(f"No load_from checkpoint in {config_path}")
    else:
        checkpoint = next(iter(sorted(exp_root.glob("best_*.pth"))), None)
        if checkpoint is None:
            raise FileNotFoundError(f"No best_*.pth checkpoint in {exp_root}")
        weights = str(checkpoint)

    stats = get_deploy_stats(str(config_path), weights, image_dir, device)
    output_path = exp_root / f"evaluation_results_{mode}_deployment.json"
    output_path.write_text(
        json.dumps({"mode": mode, "deployment": stats}, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Directory containing test images"
    )
    parser.add_argument(
        "--models", nargs="*", help="Experiment directory names; default: all"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["finetune", "zeroshot"],
        choices=("finetune", "zeroshot"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summary", default="evaluation_results_maskrcnn_deploy.json")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = {}
    experiments = sorted(path for path in args.exp_dir.iterdir() if path.is_dir())
    for exp_root in experiments:
        if args.models and exp_root.name not in args.models:
            continue
        summary[exp_root.name] = {}
        for mode in args.modes:
            print(f"Running {exp_root.name} [{mode}]")
            summary[exp_root.name][mode] = run_experiment(
                exp_root, mode, args.data_dir, args.device
            )

    summary_path = args.exp_dir / args.summary
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
