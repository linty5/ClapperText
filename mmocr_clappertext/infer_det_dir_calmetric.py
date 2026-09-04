"""Evaluate ClapperText detection runs and report deployment statistics."""

import argparse
import glob
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from mmengine.config import Config
from mmocr.apis import MMOCRInferencer
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from tqdm import tqdm


def poly_iou(poly_a: Polygon, poly_b: Polygon) -> float:
    if (not poly_a.is_valid) or (not poly_b.is_valid):
        return 0.0
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    return inter / union if union > 1e-5 else 0.0


def compute_hmean(tp: int, gt_num: int, pred_num: int):
    recall = tp / gt_num if gt_num else 0.0
    precision = tp / pred_num if pred_num else 0.0
    hmean = (
        2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
    )
    return recall, precision, hmean


def load_ground_truths_per_video(gt_dir: str) -> Dict:
    video_gt_data = {}
    for json_path in glob.glob(os.path.join(gt_dir, "*.json")):
        video = Path(json_path).stem
        video_gt_data[video] = {}
        with open(json_path, "r", encoding="utf-8") as f:
            for ln in f:
                data = json.loads(ln.strip())
                img_name = data["file_name"]
                polys = [Polygon(box["polygon"]) for box in data["gt_bboxes"]]
                video_gt_data[video][img_name] = polys
    return video_gt_data


def load_predictions(pred_dir: str) -> Dict:
    pred_data = {}
    for json_path in glob.glob(os.path.join(pred_dir, "*_ocr_results.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            for ln in f:
                data = json.loads(ln.strip())
                img_rel = "/".join(data["filename"].split("/")[-2:])
                polys, scores = [], []
                for coords, score in zip(
                    data["detections"].get("det_polygons", []),
                    data["detections"].get("det_scores", []),
                ):
                    pts = [
                        (coords[2 * i], coords[2 * i + 1])
                        for i in range(len(coords) // 2)
                    ]
                    polys.append(Polygon(pts))
                    scores.append(float(score))
                pred_data[img_rel] = {
                    "polygons": polys,
                    "scores": np.asarray(scores, dtype=np.float32),
                }
    return pred_data


def evaluate_single_video(gt_dict, pred_dict, iou_thr=0.5, score_thr=0.5):
    tot_gt = tot_pred = tot_tp = 0

    for img, gt_polys in gt_dict.items():
        pred_entry = pred_dict.get(img, {"polygons": [], "scores": []})
        pred_polys = [
            p
            for p, s in zip(pred_entry["polygons"], pred_entry["scores"])
            if s >= score_thr
        ]

        iou_mat = np.zeros((len(gt_polys), len(pred_polys)), dtype=np.float32)
        for i, g in enumerate(gt_polys):
            for j, p in enumerate(pred_polys):
                iou_mat[i, j] = poly_iou(g, p)

        if iou_mat.size:
            r, c = linear_sum_assignment(-iou_mat)
            tp = sum(iou_mat[ri, ci] >= iou_thr for ri, ci in zip(r, c))
        else:
            tp = 0

        tot_gt += len(gt_polys)
        tot_pred += len(pred_polys)
        tot_tp += tp

    return compute_hmean(tot_tp, tot_gt, tot_pred)


def evaluate_dataset(gt_dir, pred_dir, score_thr: float):
    gt_all = load_ground_truths_per_video(gt_dir)
    pred_all = load_predictions(pred_dir)

    metrics = {}
    for iou_thr in [0.5, 0.75]:
        r_l, p_l, h_l = [], [], []
        for vid, gt_vid in gt_all.items():
            r, p, h = evaluate_single_video(
                gt_vid, pred_all, iou_thr=iou_thr, score_thr=score_thr
            )
            r_l.append(r)
            p_l.append(p)
            h_l.append(h)
        metrics[f"iou_{iou_thr}"] = {
            "recall": float(np.mean(r_l)),
            "precision": float(np.mean(p_l)),
            "hmean": float(np.mean(h_l)),
        }
    return metrics


def load_score_thresholds(path: Path) -> Dict[str, float]:
    """Load the per-model operating points recorded with benchmark results."""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    thresholds = {}
    for model_name, record in records.items():
        value = record.get("score_thr") if isinstance(record, dict) else None
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(
                f"Missing or invalid score_thr for {model_name!r} in {path}"
            )
        thresholds[model_name] = float(value)
    return thresholds


def _extract_input_hw(cfg) -> tuple:
    test_pipe = None
    if hasattr(cfg, "test_pipeline"):
        test_pipe = cfg.test_pipeline
    elif hasattr(cfg, "test_dataloader") and "dataset" in cfg.test_dataloader:
        test_pipe = cfg.test_dataloader["dataset"].get("pipeline", None)

    if not test_pipe:
        return 640, 640

    for step in test_pipe:
        if not isinstance(step, dict):
            continue
        if step.get("type") == "Resize" and "scale" in step:
            scale = step["scale"]
            if isinstance(scale, tuple) and len(scale) == 2:
                return scale[1], scale[0]
        if step.get("type") == "ShortScaleAspectJitter" and "short_size" in step:
            s = step["short_size"]
            return s, s
    return 640, 640


def get_deploy_stats(det_cfg_path, det_weights, device="cuda"):
    stats = {
        "params": None,
        "gflops": None,
        "fps": None,
        "latency_ms": None,
        "mem_mb": None,
        "input_h": None,
        "input_w": None,
    }

    cfg = Config.fromfile(det_cfg_path)
    H, W = _extract_input_hw(cfg)
    raw_H, raw_W = int(H), int(W)
    H = int(np.ceil(raw_H / 32) * 32)
    W = int(np.ceil(raw_W / 32) * 32)
    stats.update(
        {"input_h": raw_H, "input_w": raw_W, "input_h_pad32": H, "input_w_pad32": W}
    )

    inferencer = MMOCRInferencer(
        det=det_cfg_path, det_weights=det_weights, device=device
    )
    model = None
    if hasattr(inferencer, "textdet_inferencer") and hasattr(
        inferencer.textdet_inferencer, "model"
    ):
        model = inferencer.textdet_inferencer.model
    elif hasattr(inferencer, "det_model"):
        model = inferencer.det_model
    elif hasattr(inferencer, "model"):
        model = inferencer.model

    if model is None:
        raise AttributeError("Cannot locate detection model inside MMOCRInferencer")

    model.eval()
    dummy = torch.randn(1, 3, H, W, device=device)
    stats["params"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)

    try:
        from thop import profile

        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        stats["gflops"] = round(flops / 1e9, 3)
    except Exception as error:
        warnings.warn(f"FLOP measurement failed: {error}")

    try:
        use_cuda = device.startswith("cuda")
        with torch.no_grad():
            for _ in range(20):
                model(dummy)
            if use_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
            repeats = 50
            start = time.perf_counter()
            for _ in range(repeats):
                model(dummy)
            if use_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            stats["fps"] = round(repeats / elapsed, 2)
            stats["latency_ms"] = round(1000 * elapsed / repeats, 2)
            if use_cuda:
                stats["mem_mb"] = round(
                    torch.cuda.max_memory_allocated(device) / 1e6, 1
                )
    except Exception as error:
        warnings.warn(f"Runtime measurement failed: {error}")

    return stats


def batch_infer(det_cfg, det_weights, data_dir, pred_dir, device="cuda", batch_size=16):
    print(f"Batch infer: {det_cfg} {det_weights} on {device}")
    inferencer = MMOCRInferencer(det=det_cfg, det_weights=det_weights, device=device)

    dirs = sorted([p for p in os.listdir(data_dir) if (Path(data_dir) / p).is_dir()])

    for sub in tqdm(dirs, desc=f"Infer [{Path(det_cfg).stem}]"):
        sub_path = Path(data_dir) / sub
        out_json = Path(pred_dir) / f"{sub}_ocr_results.json"
        imgs = sorted(
            str(sub_path / f)
            for f in os.listdir(sub_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not imgs:
            continue

        with open(out_json, "w", encoding="utf-8") as f:
            result = inferencer(imgs, batch_size=batch_size)
            for pred, img in zip(result["predictions"], imgs):
                f.write(
                    json.dumps(
                        {"filename": img, "detections": pred}, ensure_ascii=False
                    )
                    + "\n"
                )


def run_experiment(
    exp_root: Path,
    mode: str,
    data_dir: str,
    gt_dir: str,
    device: str,
    batch_size: int,
    score_thr: float,
    measure_deployment: bool,
):
    cfg_py = next(iter(sorted(exp_root.glob("*.py"))), None)
    if cfg_py is None:
        raise FileNotFoundError(f"No *.py in {exp_root}")
    best_ckpt = next(iter(sorted(exp_root.glob("best_*.pth"))), None)

    if mode == "zeroshot":
        cfg = Config.fromfile(cfg_py)
        weights = cfg.get("load_from", None)
        if not weights:
            raise ValueError(f"No load_from checkpoint in {cfg_py}")
        metric_json = exp_root / "evaluation_results_zeroshot_best.json"
        pred_dir = exp_root / "infer_results_zeroshot"
    else:
        if best_ckpt is None:
            raise FileNotFoundError(f"No best_*.pth checkpoint in {exp_root}")
        weights = str(best_ckpt)
        metric_json = exp_root / "evaluation_results_finetune_best.json"
        pred_dir = exp_root / "infer_results_finetune"

    deploy_stats = (
        get_deploy_stats(str(cfg_py), weights, device=device)
        if measure_deployment
        else {}
    )

    pred_dir.mkdir(parents=True, exist_ok=True)
    batch_infer(
        str(cfg_py),
        weights,
        data_dir,
        str(pred_dir),
        device=device,
        batch_size=batch_size,
    )

    metrics = evaluate_dataset(gt_dir, str(pred_dir), score_thr)

    out_data = {
        "mode": mode,
        "score_thr": score_thr,
        "metrics": metrics,
    }
    if deploy_stats:
        out_data["deployment"] = deploy_stats
    with open(metric_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=4)

    return metrics, deploy_stats


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
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
    parser.add_argument("--batch-size", type=int, default=4)
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument(
        "--score-thr",
        type=float,
        help="Override the score threshold for every selected model (default: 0.5)",
    )
    threshold_group.add_argument(
        "--thresholds-file",
        type=Path,
        help="Benchmark JSON containing a score_thr value for each model",
    )
    parser.add_argument(
        "--measure-deployment",
        action="store_true",
        help="Also benchmark inference speed and GPU memory",
    )
    parser.add_argument("--summary", default="evaluation_results_all.json")
    return parser.parse_args()


def main():
    args = parse_args()
    all_summary = {}
    thresholds = (
        load_score_thresholds(args.thresholds_file) if args.thresholds_file else {}
    )

    if args.score_thr is not None and not 0 <= args.score_thr <= 1:
        raise ValueError("--score-thr must be between 0 and 1")

    for exp_root in sorted(path for path in args.exp_dir.iterdir() if path.is_dir()):
        if args.models and exp_root.name not in args.models:
            continue
        if args.thresholds_file and exp_root.name not in thresholds:
            raise KeyError(
                f"No score_thr for {exp_root.name!r} in {args.thresholds_file}"
            )
        score_thr = (
            args.score_thr
            if args.score_thr is not None
            else thresholds.get(exp_root.name, 0.5)
        )
        all_summary[exp_root.name] = {}
        for mode in args.modes:
            print(f"Running {exp_root.name} [{mode}], score_thr={score_thr:g}")
            metrics, deploy_stats = run_experiment(
                exp_root,
                mode,
                str(args.data_dir),
                str(args.gt_dir),
                args.device,
                args.batch_size,
                score_thr,
                args.measure_deployment,
            )
            all_summary[exp_root.name][mode] = {
                "score_thr": score_thr,
                **metrics,
                **deploy_stats,
            }

    summary_path = args.exp_dir / args.summary
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=4)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
