import os
import json
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count


def process_file(args):
    src_path, dst_path = args
    with open(src_path, "r") as f:
        data = json.load(f)
    data.pop("hemodynamics", None)
    with open(dst_path, "w") as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser()
    base_dir = Path(__file__).parent / "train"
    parser.add_argument("--src_dir", default=str(base_dir / "synthetic_dataset_aug10x_hemo"))
    parser.add_argument("--dst_dir", default=str(base_dir / "synthetic_dataset_aug10x"))
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for src_path in src_dir.glob("*_hemo.json"):
        dst_name = src_path.name.replace("_hemo.json", ".json")
        tasks.append((src_path, dst_dir / dst_name))

    print(f"Found {len(tasks)} files in {src_dir}")
    print(f"Output → {dst_dir}")

    with Pool(args.workers) as pool:
        pool.map(process_file, tasks, chunksize=200)

    print(f"Done. {len(tasks)} files written.")


if __name__ == "__main__":
    main()
