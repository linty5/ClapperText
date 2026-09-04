"""Run text recognition over per-video word-crop folders with MMOCR."""

import argparse
import json
import time
from pathlib import Path

from mmocr.apis import MMOCRInferencer
from tqdm import tqdm

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def batch_infer(
    input_dir, output_dir, model, weights=None, device="cuda", batch_size=64
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inferencer = MMOCRInferencer(rec=str(model), rec_weights=weights, device=device)

    video_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    start = time.time()
    for video_dir in tqdm(video_dirs, desc=Path(str(model)).stem):
        images = sorted(
            str(path)
            for path in video_dir.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        output_path = output_dir / f"{video_dir.name}_ocr_results.json"
        result = inferencer(images, batch_size=batch_size)
        with output_path.open("w", encoding="utf-8") as output_file:
            for prediction, image_path in zip(result["predictions"], images):
                record = {"filename": image_path, "predictions": prediction}
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Finished in {time.time() - start:.2f} seconds")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--model", required=True, help="MMOCR model name or config path"
    )
    parser.add_argument("--weights", help="Optional checkpoint path or URL")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    batch_infer(
        args.input_dir,
        args.output_dir,
        args.model,
        args.weights,
        args.device,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
