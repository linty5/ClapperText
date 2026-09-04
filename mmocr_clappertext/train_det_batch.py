"""Train one or more ClapperText detection configurations with MMOCR."""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs", nargs="+", type=Path, help="MMOCR config files to train"
    )
    parser.add_argument("--work-root", type=Path, default=Path("train_logs/det"))
    parser.add_argument("--train-script", type=Path, default=Path("tools/train.py"))
    parser.add_argument(
        "--launcher", default="none", choices=("none", "pytorch", "slurm", "mpi")
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--auto-scale-lr", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--keep-going", action="store_true", help="Continue after a failed training run"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    missing = [str(path) for path in args.configs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Config files not found: {', '.join(missing)}")
    if not args.train_script.is_file():
        raise FileNotFoundError(f"Training script not found: {args.train_script}")

    failed = []
    for config_path in args.configs:
        work_dir = args.work_root / config_path.stem
        command = [
            sys.executable,
            str(args.train_script),
            str(config_path),
            "--work-dir",
            str(work_dir),
            "--launcher",
            args.launcher,
        ]
        if args.amp:
            command.append("--amp")
        if args.auto_scale_lr:
            command.append("--auto-scale-lr")
        if args.resume:
            command.append("--resume")

        print(f"Training {config_path.stem}")
        start = time.time()
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            failed.append(config_path)
            if not args.keep_going:
                raise
        else:
            print(f"Finished in {(time.time() - start) / 60:.2f} minutes")

    if failed:
        names = ", ".join(path.stem for path in failed)
        raise SystemExit(f"Training failed for: {names}")


if __name__ == "__main__":
    main()
