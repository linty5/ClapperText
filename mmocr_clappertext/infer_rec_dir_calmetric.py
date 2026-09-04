"""Evaluate ClapperText recognition runs with per-video metrics."""

import argparse
import csv
import json
import re
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import Levenshtein
import torch
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmocr.apis import MMOCRInferencer
from tqdm import tqdm

_SYMBOL_RE = re.compile(r"[^A-Za-z0-9\u4e00-\u9fa5]")


def norm_text(txt: str, mode: str):
    """mode: raw | ignore_case | ignore_case_symbol"""
    if mode == "ignore_case":
        return txt.lower()
    if mode == "ignore_case_symbol":
        return _SYMBOL_RE.sub("", txt).lower()
    return txt


def word_accuracy(pairs, mode="raw"):
    ok = sum(norm_text(p, mode) == norm_text(g, mode) for g, p in pairs)
    return ok / len(pairs) if pairs else 0.0


def cer_wer(pairs):
    total_chars = total_err_char = 0
    total_words = total_err_word = 0
    for gt, pred in pairs:
        g = norm_text(gt, "ignore_case_symbol")
        p = norm_text(pred, "ignore_case_symbol")
        if g == "":
            continue
        total_words += 1
        total_err_word += int(g != p)
        total_chars += len(g)
        total_err_char += Levenshtein.distance(g, p)
    cer = total_err_char / total_chars if total_chars else 0
    wer = total_err_word / total_words if total_words else 0
    return cer, wer


# ---------- Ground-truth loader ---------- #
def reconstruct_occluded(text, occ_text):
    out = list(text)
    occ = list(occ_text)
    for i, c in enumerate(out):
        if c == "?" and occ:
            out[i] = occ.pop(0)
    return "".join(out)


def load_gt_from_csv_dir(
    gt_dir: str,
    include_occluded: bool,
    handwritten_filter=None,
    occluded_filter=None,
    category_filter=None,
) -> Dict[str, tuple]:
    """
    Parse every *.csv in gt_dir and return:
        filename → (text, handwritten_flag, occluded_flag)
    """
    gt = {}
    for csv_path in Path(gt_dir).glob("*.csv"):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["Filename"].strip()
                occ = int(row["Occluded"])
                hw = int(row["Handwritten"])
                cat = row.get("Category", "")

                if not include_occluded and occ == 1:
                    continue
                if handwritten_filter is not None and hw != handwritten_filter:
                    continue
                if occluded_filter is not None and occ != occluded_filter:
                    continue
                if category_filter and cat not in category_filter:
                    continue

                txt = row["Text"].strip()
                if occ == 1:
                    txt = reconstruct_occluded(txt, row["Occluded_Text"].strip())
                gt[fname] = (txt, hw, occ)
    return gt


def metrics_for_video(pred_json: str, gt: dict):
    pairs_all = []
    pairs_hw = []
    pairs_nhw = []
    with open(pred_json, "r", encoding="utf-8") as f:
        for ln in f:
            rec = json.loads(ln.strip())
            fname = Path(rec["filename"]).name
            if fname not in gt:
                continue
            pred_txt = rec["predictions"]["rec_texts"][0]
            gt_txt, hw_flag, _ = gt[fname]
            pairs_all.append((gt_txt, pred_txt))
            (pairs_hw if hw_flag == 1 else pairs_nhw).append((gt_txt, pred_txt))

    def aggregate(pairs):
        if not pairs:
            return None
        cer, wer = cer_wer(pairs)
        return {
            "word_acc": word_accuracy(pairs, "raw"),
            "word_acc_ignore_case": word_accuracy(pairs, "ignore_case"),
            "word_acc_ignore_case_symbol": word_accuracy(pairs, "ignore_case_symbol"),
            "cer": cer,
            "wer": wer,
        }

    return {
        "overall": aggregate(pairs_all),
        "handwritten": aggregate(pairs_hw),
        "non_handwritten": aggregate(pairs_nhw),
    }


def average_metrics(metrics_list: List[Dict]):
    """
    Macro-average over videos, but skip videos where a split is None.
    """
    acc_sum = defaultdict(lambda: defaultdict(float))
    acc_cnt = defaultdict(lambda: defaultdict(int))

    for vid_metrics in metrics_list:
        for (
            split,
            vals,
        ) in vid_metrics.items():  # overall / handwritten / non_handwritten
            if vals is None:
                continue
            for k, v in vals.items():
                acc_sum[split][k] += v
                acc_cnt[split][k] += 1

    out = {}
    for split in acc_sum:
        out[split] = {}
        for k in acc_sum[split]:
            n = acc_cnt[split][k]
            out[split][k] = acc_sum[split][k] / n if n else 0.0
    return out


def batch_infer_rec(rec_cfg, rec_weights, img_root, out_dir, device="cuda", bs=64):
    out_dir.mkdir(parents=True, exist_ok=True)
    infer = MMOCRInferencer(rec=rec_cfg, rec_weights=rec_weights, device=device)

    for vid in tqdm(
        sorted([p for p in img_root.iterdir() if p.is_dir()]), desc=f"Infer [{rec_cfg}]"
    ):
        out_json = out_dir / f"{vid.name}_ocr_results.json"
        imgs = sorted(
            str(f)
            for f in vid.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        )
        if not imgs:
            continue
        with open(out_json, "w", encoding="utf-8") as f:
            res = infer(imgs, batch_size=bs)
            for pred, im in zip(res["predictions"], imgs):
                f.write(
                    json.dumps(
                        {"filename": im, "predictions": pred}, ensure_ascii=False
                    )
                    + "\n"
                )


def _extract_rec_input_hw(cfg) -> tuple:
    pipe = None
    if hasattr(cfg, "test_pipeline"):
        pipe = cfg.test_pipeline
    elif hasattr(cfg, "test_dataloader"):
        pipe = cfg.test_dataloader["dataset"].get("pipeline", None)

    H, W = 32, 160  # sensible defaults
    if not pipe:
        return H, W

    for step in pipe:
        if not isinstance(step, dict):
            continue
        tp = step.get("type")

        if tp == "RescaleToHeight":
            H = step.get("height", H)
            mw = step.get("max_width", step.get("width", None))
            if mw is not None:
                W = mw

        elif tp == "Resize" and "scale" in step:
            sc = step["scale"]  # (W, H)
            if isinstance(sc, (tuple, list)) and len(sc) == 2:
                W, H = sc

        elif tp == "PadToWidth":
            W = max(W, step.get("width", W))

    if W is None:
        W = max(160, H * 5)

    return int(H), int(W)


def _extract_rec_input_c(hw_cfg) -> int:
    if hasattr(hw_cfg, "model") and hasattr(hw_cfg.model, "data_preprocessor"):
        mp = hw_cfg.model.data_preprocessor
        if isinstance(mp.get("mean", None), (list, tuple)):
            return len(mp["mean"])
    pipe = None
    if hasattr(hw_cfg, "test_pipeline"):
        pipe = hw_cfg.test_pipeline
    elif hasattr(hw_cfg, "test_dataloader"):
        pipe = hw_cfg.test_dataloader["dataset"].get("pipeline", None)

    if pipe:
        for st in pipe:
            if isinstance(st, dict) and st.get("type") == "LoadImageFromFile":
                ct = st.get("color_type", "").lower()
                if "gray" in ct:
                    return 1
    return 3  # default RGB


def get_deploy_stats_rec(rec_cfg_path, rec_weights, device="cuda", bs_deploy=128):
    stats = {
        "params": None,
        "fps": None,
        "latency_ms": None,
        "mem_mb": None,
        "input_h": None,
        "input_w": None,
        "channels": None,
    }

    cfg = Config.fromfile(rec_cfg_path)
    H, W = _extract_rec_input_hw(cfg)
    C = _extract_rec_input_c(cfg)
    stats.update({"input_h": H, "input_w": W, "channels": C})

    dummy = torch.randn(bs_deploy, C, H, W, device=device)
    dummy_data_samples = [InstanceData() for _ in range(bs_deploy)]

    try:
        infer = MMOCRInferencer(
            rec=rec_cfg_path,
            rec_weights=rec_weights,
            device=device,
        )

        model = infer.textrec_inferencer.model
        model.eval()
        stats["params"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)

        use_cuda = device.startswith("cuda")
        with torch.no_grad():
            if use_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(5):
                model(dummy, data_samples=dummy_data_samples)
            if use_cuda:
                torch.cuda.synchronize()
            repeats = 30
            start = time.perf_counter()
            for _ in range(repeats):
                model(dummy, data_samples=dummy_data_samples)
            if use_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            stats["fps"] = round(bs_deploy * repeats / elapsed, 2)
            stats["latency_ms"] = round(1000 * elapsed / repeats / bs_deploy, 2)
            if use_cuda:
                stats["mem_mb"] = round(
                    torch.cuda.max_memory_allocated(device) / 1e6, 1
                )

    except Exception as e:
        warnings.warn(f"Deployment measurement failed: {e}")
        stats["fps"] = None
        stats["latency_ms"] = None
        stats["mem_mb"] = None

    return stats


def run_rec_experiment(
    exp_root: Path,
    mode: str,
    img_root: Path,
    gt_dir: str,
    device: str,
    batch_size: int,
    deployment_batch_size: int,
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
        metric_json = exp_root / "rec_eval_zeroshot.json"
        pred_dir = exp_root / "rec_preds_zeroshot"
    else:
        if best_ckpt is None:
            raise FileNotFoundError(f"No best_*.pth checkpoint in {exp_root}")
        weights = str(best_ckpt)
        metric_json = exp_root / "rec_eval_finetune.json"
        pred_dir = exp_root / "rec_preds_finetune"

    deploy_stats = (
        get_deploy_stats_rec(
            str(cfg_py), weights, device=device, bs_deploy=deployment_batch_size
        )
        if measure_deployment
        else None
    )

    batch_infer_rec(
        str(cfg_py), weights, img_root, pred_dir, device=device, bs=batch_size
    )

    gt_excl = load_gt_from_csv_dir(gt_dir, include_occluded=False)
    gt_incl = load_gt_from_csv_dir(gt_dir, include_occluded=True)
    gt_occ = load_gt_from_csv_dir(gt_dir, include_occluded=True, occluded_filter=1)

    metrics_excl = [
        metrics_for_video(j, gt_excl) for j in pred_dir.glob("*_ocr_results.json")
    ]
    metrics_incl = [
        metrics_for_video(j, gt_incl) for j in pred_dir.glob("*_ocr_results.json")
    ]
    metrics_occ = [
        metrics_for_video(j, gt_occ) for j in pred_dir.glob("*_ocr_results.json")
    ]

    out = {
        "mode": mode,
        "exclude_occluded": average_metrics(metrics_excl),
        "include_occluded": average_metrics(metrics_incl),
        "only_occluded": average_metrics(metrics_occ),
    }
    if deploy_stats is not None:
        out["deployment"] = deploy_stats

    with open(metric_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="*", help="Experiment directory names; default: all"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["zeroshot", "finetune"],
        choices=("zeroshot", "finetune"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--measure-deployment",
        action="store_true",
        help="Also benchmark inference speed and GPU memory",
    )
    parser.add_argument("--deployment-batch-size", type=int, default=128)
    parser.add_argument("--summary", default="evaluation_results_recognition.json")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = {}
    for exp_root in sorted(path for path in args.exp_dir.iterdir() if path.is_dir()):
        if args.models and exp_root.name not in args.models:
            continue
        summary[exp_root.name] = {}
        for mode in args.modes:
            print(f"Running {exp_root.name} [{mode}]")
            result = run_rec_experiment(
                exp_root,
                mode,
                args.image_root,
                str(args.gt_dir),
                args.device,
                args.batch_size,
                args.deployment_batch_size,
                args.measure_deployment,
            )
            summary[exp_root.name][mode] = result
    with open(args.exp_dir / args.summary, "w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=4)
    print("Done")


if __name__ == "__main__":
    main()
