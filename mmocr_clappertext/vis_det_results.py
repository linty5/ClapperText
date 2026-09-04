"""Draw MMOCR detection predictions on ClapperText frames."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def draw_detections(image_path, detections):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    polygons = detections.get("det_polygons", [])
    scores = detections.get("det_scores", [])
    for index, polygon in enumerate(polygons):
        points = np.asarray(polygon, dtype=np.int32).reshape((-1, 2))
        cv2.polylines(image, [points], True, (0, 255, 0), 2)
        score = f"{scores[index]:.2f}" if index < len(scores) else "N/A"
        cv2.putText(
            image,
            score,
            tuple(points[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    return image


def visualize(json_dir, image_dir, output_dir):
    json_dir = Path(json_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    for json_path in tqdm(sorted(json_dir.glob("*.json"))):
        suffix = "_ocr_results"
        video_name = json_path.stem
        if video_name.endswith(suffix):
            video_name = video_name[: -len(suffix)]
        video_output = output_dir / video_name
        video_output.mkdir(parents=True, exist_ok=True)

        with json_path.open(encoding="utf-8") as input_file:
            for line in input_file:
                record = json.loads(line)
                image_name = Path(record["filename"]).name
                image_path = image_dir / video_name / image_name
                rendered = draw_detections(image_path, record["detections"])
                cv2.imwrite(str(video_output / image_name), rendered)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_dir", type=Path, help="Directory containing per-video prediction JSON"
    )
    parser.add_argument(
        "image_dir", type=Path, help="Directory containing per-video image folders"
    )
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    visualize(args.json_dir, args.image_dir, args.output_dir)


if __name__ == "__main__":
    main()
