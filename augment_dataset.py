import os
import json
import glob
import random
import copy
from pathlib import Path
import argparse

def augment_dataset(input_dir: str, output_dir: str, num_samples: int = 10, seed: int = 42):
    random.seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    json_files = glob.glob(os.path.join(input_path, '**', '*.json'), recursive=True)
    print(f"Starting data augmentation...")
    print(f"Found {len(json_files)} JSON source files in {input_dir}")
    print(f"Augmentation target: each file will be augmented {num_samples}x -> expected {len(json_files) * num_samples} files.")
    print("-" * 50)

    for count, filepath in enumerate(json_files):
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Failed to parse {filepath}: {e}")
                continue

        bc = data.get("boundary_conditions", {})
        pressure_range = bc.get("pressure", 100.0)
        v_range = bc.get("velocity_range", [1.0, 1.0])

        rel_path = os.path.relpath(filepath, input_dir)
        filename = os.path.basename(rel_path)
        name, ext = os.path.splitext(filename)

        out_file_dir = output_path
        out_file_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            if isinstance(pressure_range, list) and len(pressure_range) == 2:
                sampled_p = random.uniform(pressure_range[0], pressure_range[1])
            else:
                sampled_p = float(pressure_range)

            if isinstance(v_range, list) and len(v_range) == 2:
                sampled_v = random.uniform(v_range[0], v_range[1])
            elif isinstance(v_range, (int, float)):
                sampled_v = float(v_range)
            else:
                sampled_v = 1.0

            aug_data = copy.deepcopy(data)

            aug_data["boundary_conditions"]["sampled_pressure"] = sampled_p
            aug_data["boundary_conditions"]["sampled_velocity"] = sampled_v

            original_case_id = aug_data.get("case_id", name)
            new_case_id = f"{original_case_id}_bc{i}"
            aug_data["case_id"] = new_case_id

            new_filename = f"{name}_bc{i}{ext}"
            new_filepath = out_file_dir / new_filename

            with open(new_filepath, 'w') as out_f:
                json.dump(aug_data, out_f, indent=2)

        if (count + 1) % 50 == 0:
            print(f"Processed {count + 1} / {len(json_files)} source files...")

    print("-" * 50)
    print(f"Static data augmentation complete!\nResults saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vascular JSON Dataset Static Augmentation Tool")
    parser.add_argument("--input_dir", type=str, default=r"train/synthetic_dataset", help="Original dataset root directory")
    parser.add_argument("--output_dir", type=str, default=r"train/synthetic_dataset_aug10x", help="Target root directory to store 10x augmented files")
    parser.add_argument("--samples", type=int, default=10, help="Number of bounding conditions to sample per file")

    args = parser.parse_args()

    augment_dataset(args.input_dir, args.output_dir, num_samples=args.samples)
