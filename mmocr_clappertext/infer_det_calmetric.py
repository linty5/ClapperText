"""Run one ClapperText detector and compute per-video Hmean."""

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
from mmengine.config import Config
from mmocr.apis import MMOCRInferencer
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from tqdm import tqdm


def batch_infer_mmocr(
    det_path, weights_path, data_dir, pred_dir, device="cuda", batch_size=16
):
    os.makedirs(pred_dir, exist_ok=True)
    inferencer = MMOCRInferencer(det=det_path, device=device, det_weights=weights_path)

    dir_list = sorted(os.listdir(data_dir))

    start_time = time.time()

    for subdir in tqdm(dir_list):
        subdir_path = os.path.join(data_dir, subdir)

        if not os.path.isdir(subdir_path):
            continue

        result_file = os.path.join(pred_dir, f"{subdir}_ocr_results.json")
        image_files = sorted(
            os.path.join(subdir_path, f)
            for f in os.listdir(subdir_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not image_files:
            continue

        with open(result_file, "w", encoding="utf-8") as f:
            result = inferencer(image_files, batch_size=batch_size)

            predictions = result.get("predictions", [])

            for prediction, image_path in zip(predictions, image_files):
                json_line = json.dumps(
                    {"filename": image_path, "detections": prediction},
                    ensure_ascii=False,
                )
                f.write(json_line + "\n")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Batch inference completed in {total_time:.2f} seconds")


def poly_iou(poly_a: Polygon, poly_b: Polygon) -> float:
    if not poly_a.is_valid or not poly_b.is_valid:
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


def load_ground_truths_per_video(gt_dir: str):
    video_gt_data = {}
    for json_path in glob.glob(os.path.join(gt_dir, "*.json")):
        video_name = os.path.splitext(os.path.basename(json_path))[0]
        video_gt_data[video_name] = {}
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                img_name = data["file_name"]
                polygons = [Polygon(box["polygon"]) for box in data["gt_bboxes"]]
                video_gt_data[video_name][img_name] = polygons
    return video_gt_data


def load_predictions(pred_dir: str):
    pred_data = {}
    for json_path in glob.glob(os.path.join(pred_dir, "*_ocr_results.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                name_list = data["filename"].split("/")[-2:]
                img_name = "/".join(name_list)
                polygons = []
                scores = []
                for poly_coords, score in zip(
                    data["detections"].get("det_polygons", []),
                    data["detections"].get("det_scores", []),
                ):
                    pts = [
                        (poly_coords[2 * i], poly_coords[2 * i + 1])
                        for i in range(len(poly_coords) // 2)
                    ]
                    polygons.append(Polygon(pts))
                    scores.append(float(score))
                pred_data[img_name] = {
                    "polygons": polygons,
                    "scores": np.array(scores, dtype=np.float32),
                }
    return pred_data


def evaluate_single_video(gt_dict, pred_dict, iou_thr=0.5, score_thr=0.5):
    all_gt = []
    all_pred = []

    for img_name, gt_polygons in gt_dict.items():
        pred_entry = pred_dict.get(img_name, {"polygons": [], "scores": []})
        pred_polygons = [
            poly
            for poly, score in zip(pred_entry["polygons"], pred_entry["scores"])
            if score >= score_thr
        ]
        all_gt.append(gt_polygons)
        all_pred.append(pred_polygons)

    total_gt = 0
    total_pred = 0
    total_tp = 0

    for gt_polys, pred_polys in zip(all_gt, all_pred):
        iou_matrix = np.zeros((len(gt_polys), len(pred_polys)), dtype=np.float32)
        for i, gt_poly in enumerate(gt_polys):
            for j, pred_poly in enumerate(pred_polys):
                iou_matrix[i, j] = poly_iou(gt_poly, pred_poly)

        if iou_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            tp = sum(iou_matrix[r, c] >= iou_thr for r, c in zip(row_ind, col_ind))
        else:
            tp = 0

        total_gt += len(gt_polys)
        total_pred += len(pred_polys)
        total_tp += tp

    return compute_hmean(total_tp, total_gt, total_pred)


def evaluate_all(gt_dir, pred_dir, metric_path, iou_thr=0.5, score_thr=0.5):
    gt_data = load_ground_truths_per_video(gt_dir)
    pred_data = load_predictions(pred_dir)

    video_metrics = {}
    recall_list, precision_list, hmean_list = [], [], []

    for video, gt_video_data in tqdm(gt_data.items()):
        recall, precision, hmean = evaluate_single_video(
            gt_video_data, pred_data, iou_thr, score_thr
        )
        video_metrics[video] = {
            "recall": recall,
            "precision": precision,
            "hmean": hmean,
        }
        recall_list.append(recall)
        precision_list.append(precision)
        hmean_list.append(hmean)

    summary = {
        "video_metrics": video_metrics,
        "average": {
            "recall": float(np.mean(recall_list)),
            "precision": float(np.mean(precision_list)),
            "hmean": float(np.mean(hmean_list)),
        },
    }
    print(summary["average"])

    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    print(f"Evaluation completed. Saved to {metric_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("gt_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--weights", help=("Checkpoint path or URL; defaults to config.load_from")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--score-thr", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    weights = args.weights or Config.fromfile(args.config).get("load_from")
    if not weights:
        raise ValueError("No checkpoint was provided and config.load_from is empty")

    prediction_dir = args.output_dir / "predictions"
    metric_path = args.output_dir / "metrics.json"
    batch_infer_mmocr(
        str(args.config),
        weights,
        str(args.data_dir),
        str(prediction_dir),
        args.device,
        args.batch_size,
    )
    evaluate_all(
        str(args.gt_dir),
        str(prediction_dir),
        str(metric_path),
        iou_thr=args.iou_thr,
        score_thr=args.score_thr,
    )


if __name__ == "__main__":
    main()
