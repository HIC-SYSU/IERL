import os
import re
import json
import glob
import numpy as np
from typing import List, Dict, Any, Iterator
from torch.utils.data import Dataset
import random
from collections import defaultdict


def generate_adaptive_grid(x_true: np.ndarray, r_true: np.ndarray, L: int = 100, alpha: float = 8.0) -> np.ndarray:
    dx = np.gradient(x_true)
    dr = np.gradient(r_true)
    grad = np.abs(dr / (dx + 1e-6))
    density = 1.0 + alpha * grad

    cdf = np.zeros_like(x_true, dtype=np.float64)
    for i in range(1, len(x_true)):
        cdf[i] = cdf[i - 1] + 0.5 * (density[i] + density[i - 1]) * (x_true[i] - x_true[i - 1])

    cdf_norm = cdf / (cdf[-1] + 1e-12)
    p_uniform = np.linspace(0.0, 1.0, L)
    x_grid = np.interp(p_uniform, cdf_norm, x_true)
    return x_grid.astype(np.float32)


def calculate_cumulative_length(pos_list: np.ndarray) -> np.ndarray:
    pos = np.array(pos_list)
    diffs = np.diff(pos, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate(([0.0], np.cumsum(dists)))
    return cum_dists

def extract_main_branch(segments: List[Dict]) -> Dict[str, np.ndarray]:
    seg_dict = {seg.get('seg_id', i): seg for i, seg in enumerate(segments)}

    roots = [seg for seg in segments if seg.get('type') == 'Root' or seg.get('parent') is None]
    if not roots:
        roots = [segments[0]]

    current_seg = roots[0]
    main_branch_x = []
    main_branch_d = []
    main_branch_ref_d = []
    main_branch_pos = []

    current_x_offset = 0.0

    while True:
        pos = np.array(current_seg['pos'])
        radius = np.array(current_seg['radius'])
        ref_radius = np.array(current_seg['ref_radius'])

        seg_x = calculate_cumulative_length(pos) + current_x_offset

        if len(main_branch_x) > 0:
            seg_x = seg_x[1:]
            radius = radius[1:]
            ref_radius = ref_radius[1:]
            pos = pos[1:]

        main_branch_x.extend(seg_x)
        main_branch_d.extend(2.0 * radius)
        main_branch_ref_d.extend(2.0 * ref_radius)
        main_branch_pos.extend(pos)

        current_x_offset = main_branch_x[-1] if len(main_branch_x) > 0 else 0.0

        next_seg = None
        for child_id in current_seg.get('children', []):
            if child_id in seg_dict and seg_dict[child_id].get('type') in ['Main', 'Root']:
                next_seg = seg_dict[child_id]
                break

        if next_seg is None:
            break

        current_seg = next_seg

    return {
        'x': np.array(main_branch_x, dtype=np.float32),
        'd': np.array(main_branch_d, dtype=np.float32),
        'ref_d': np.array(main_branch_ref_d, dtype=np.float32),
        'pos': np.array(main_branch_pos, dtype=np.float32)
    }

def get_reference_strategy(x: np.ndarray, d: np.ndarray, ref_d: np.ndarray, stenosis_labels: list = None) -> np.ndarray:
    if stenosis_labels and len(stenosis_labels) > 0:
        start_mms = [label.get('global_start_mm', 0.0) for label in stenosis_labels]
        end_mms = [label.get('global_end_mm', 0.0) for label in stenosis_labels]

        global_start = min(start_mms)
        global_end = max(end_mms)

        margin = 2.0
        stent_start = max(0.0, global_start - margin / 2.0)
        stent_end = min(x[-1], global_end + margin / 2.0)

        ref_pos = (stent_start + stent_end) / 2.0
        ref_len = stent_end - stent_start

        mask = (x >= stent_start) & (x <= stent_end)
        if np.any(mask):
            ref_size = float(np.max(ref_d[mask]))
        else:
            ref_size = float(np.median(ref_d))

        return np.array([ref_pos, ref_len, ref_size], dtype=np.float32)

    diff = ref_d - d
    stenosis_mask = diff > (0.1 * ref_d)

    if not np.any(stenosis_mask):
        return np.array([x[-1]/2, 10.0, float(np.median(ref_d))], dtype=np.float32)

    indices = np.where(stenosis_mask)[0]

    start_idx = indices[0]
    end_idx = indices[-1]

    start_idx = max(0, start_idx - 2)
    end_idx = min(len(x) - 1, end_idx + 2)

    ref_pos = (x[start_idx] + x[end_idx]) / 2.0
    ref_len = x[end_idx] - x[start_idx] + 2.0
    ref_size = float(np.max(ref_d[start_idx:end_idx+1]))

    return np.array([ref_pos, ref_len, ref_size], dtype=np.float32)

class VascularDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", val_ratio: float = 0.2, seed: int = 42, seq_len: int = 100):
        self.seq_len = seq_len
        self.data_dir = data_dir
        all_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.json'), recursive=True))
        all_files = [f for f in all_files if os.path.basename(f) != "data_split_manifest.json"]

        def get_original_id(filepath):
            name = os.path.splitext(os.path.basename(filepath))[0]
            name = re.sub(r'_hemo$', '', name)
            name = re.sub(r'_bc\d+$', '', name)
            return name

        groups = defaultdict(list)
        for f in all_files:
            groups[get_original_id(f)].append(f)

        original_ids = sorted(groups.keys())
        rng = np.random.default_rng(seed)
        shuffled_ids = [original_ids[i] for i in rng.permutation(len(original_ids))]

        n_val = int(len(original_ids) * val_ratio)
        val_id_set   = set(shuffled_ids[:n_val])
        train_id_set = set(shuffled_ids[n_val:])

        if split == "val":
            self.valid_files = [f for oid, files in groups.items() if oid in val_id_set   for f in files]
        else:
            self.valid_files = [f for oid, files in groups.items() if oid in train_id_set for f in files]

        print(f"[VascularDataset] split='{split}' | {len(self.valid_files)} files / "
              f"{len(val_id_set) if split=='val' else len(train_id_set)} original vessels "
              f"(total {len(original_ids)} vessels, {len(all_files)} files)")

        if split == "train":
            manifest_path = os.path.join(os.path.dirname(data_dir), "data_split_manifest.json")
            manifest = {
                "seed": int(seed),
                "val_ratio": val_ratio,
                "total_vessels": len(original_ids),
                "total_files": len(all_files),
                "train_vessels": sorted(train_id_set),
                "val_vessels":   sorted(val_id_set),
            }
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"[VascularDataset] Split manifest saved → {manifest_path}")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filepath = self.valid_files[idx]
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            if 'segments' not in data:
                raise ValueError("No segments found in JSON")

            branch_data = extract_main_branch(data['segments'])

            x = branch_data['x']
            x_raw = branch_data['x']
            d_raw = branch_data['d']
            ref_d = branch_data['ref_d']

            if len(x_raw) < 5:
                raise ValueError("Vessel too short")

            x_grid = generate_adaptive_grid(x_raw, d_raw / 2.0, L=self.seq_len, alpha=8.0)
            d_grid = np.interp(x_grid, x_raw, d_raw).astype(np.float32)

            stenosis_labels = data.get('stenosis_labels', None)
            ref_stent = get_reference_strategy(x_raw, d_raw, ref_d, stenosis_labels=stenosis_labels)

            vessel_sample = {
                "x_raw":     x_raw,
                "d_raw":     d_raw,
                "x_grid":    x_grid,
                "d_grid":    d_grid,
                "x":         x_grid,
                "d":         d_grid,
                "ref_stent": ref_stent,
                "tree_dict": data,
                "patient":   data.get("case_id", os.path.basename(filepath).replace('.json', ''))
            }

            hemo = data.get("hemodynamics", None)
            if hemo is not None and "ffr_main" in hemo:
                vessel_sample["measured_ffr"] = float(hemo.get("measured_ffr", 1.0))
                vessel_sample["ffr"]          = np.asarray(hemo["ffr_main"],      dtype=np.float32)
                vessel_sample["pressure"]     = np.asarray(hemo["pressure_main"], dtype=np.float32)
                vessel_sample["flow"]         = np.asarray(hemo["flow_main"],     dtype=np.float32)
                vessel_sample["x_precomp"]    = np.asarray(hemo["x_main"],        dtype=np.float32)

            measure_site = data.get("measure_site", {})
            vessel_sample["measure_site_mm"] = float(measure_site.get("global_pos_mm", -1.0))

            return vessel_sample
        except Exception as e:
            print(f"Error loading {filepath}: {e}. Retrying another sample...")
            return self.__getitem__((idx + 1) % len(self.valid_files))

class VascularDataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.current_index = 0
        if self.shuffle:
            random.shuffle(self.indices)

    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        self.current_index = 0
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self) -> List[Dict[str, Any]]:
        if self.current_index >= len(self.dataset):
            raise StopIteration

        start = self.current_index
        end = min(start + self.batch_size, len(self.dataset))
        batch_indices = self.indices[start:end]

        batch = [self.dataset[i] for i in batch_indices]
        self.current_index = end

        return batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
