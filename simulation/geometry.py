import numpy as np
import json
import math
import warnings
from typing import List, Optional
from scipy.optimize import curve_fit, OptimizeWarning
import pwlf

warnings.simplefilter("ignore", OptimizeWarning)

__all__ = ["Geometry", "Geometry_SynData"]


class Geometry:

    def __init__(self, name: str, scale: float = 0.0833):
        self.name = name
        self.segments: list = []
        self.links: list = []
        self.radii: list = []
        self.ref_radii: list = []
        self.conditions: list = []
        self.labels = None
        self.scale = scale
        self.bifur_index = None
        self.virbnum = 10

    def set_segnum(
        self,
        seg_size: float,
        seg_min_num: int = 3,
        seg_max_num: int = 35,
    ) -> None:
        self.seg_size = seg_size
        self.seg_min_num = seg_min_num
        self.segments_segnum = []
        for segment in self.segments:
            seg_len = segment[-1][0] - segment[0][0]
            num_seg = int(seg_len / seg_size)
            num_seg = max(seg_min_num, min(seg_max_num, num_seg))
            self.segments_segnum.append(num_seg)

    def self_adaptive_segments(
        self,
        thinning_strategy: str = "pwlf",
        stenosis_threshold: float = 0.1,
    ) -> None:
        self.stenosis_threshold = stenosis_threshold
        self.ori_segments = self.segments.copy()
        self.sample_1d_indexs = []

        resample_segments, resample_radii = [], []
        resample_ref_radii, resample_lengths = [], []
        segs_stenosis_value = []

        for seg_id, (seg, radius, ref_radius) in enumerate(
            zip(self.segments, self.radii, self.ref_radii)
        ):
            path_1d = [p[0] for p in seg]

            if seg_id % 2 == 0:
                area = [math.pi * r ** 2 for r in radius]
                num_seg = self.segments_segnum[seg_id]

                if thinning_strategy == "pwlf":
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            path_pwlf = path_1d.copy()
                            for i in range(1, len(path_pwlf)):
                                if path_pwlf[i] <= path_pwlf[i-1]:
                                    path_pwlf[i] = path_pwlf[i-1] + 1e-6

                            fitter = pwlf.PiecewiseLinFit(path_pwlf, area)
                            sample_1d = fitter.fitfast(num_seg, pop=3).tolist()
                            sample_idx = [_closest_index(path_1d, p) for p in sample_1d]
                            if _has_duplicates(sample_idx):
                                num_seg += 1
                                sample_1d = fitter.fitfast(num_seg, pop=3).tolist()
                                sample_idx = sorted(set(
                                    _closest_index(path_1d, p) for p in sample_1d
                                ))
                    except Exception:
                        sample_idx = [
                            int(i * (len(path_1d) - 1) / num_seg)
                            for i in range(num_seg + 1)
                        ]
                else:
                    sample_idx = [
                        int(i * (len(path_1d) - 1) / num_seg)
                        for i in range(num_seg + 1)
                    ]

                self.sample_1d_indexs.append(sample_idx)
                x_sample = [path_1d[i] for i in sample_idx]
                lengths = np.diff(x_sample).tolist()

                stenosis_vals = []
                for i in range(len(sample_idx) - 1):
                    lo, hi = sample_idx[i], sample_idx[i + 1]
                    sub_r = radius[lo: hi + 1]
                    min_idx = int(np.argmin(sub_r))
                    ref_r = ref_radius[lo + min_idx]
                    stenosis_vals.append(
                        max(1 - sub_r[min_idx] / ref_r, 0) if ref_r > 0 else 0
                    )

                segs_stenosis_value.append(stenosis_vals)
                resample_segments.append([[x, 0] for x in x_sample])
                resample_lengths.append(lengths)
                resample_radii.append([radius[i] for i in sample_idx])
                resample_ref_radii.append([ref_radius[i] for i in sample_idx])

            else:
                segs_stenosis_value.append([0] * len(seg))
                resample_segments.append(seg)
                resample_lengths.append(np.diff(path_1d).tolist())
                resample_radii.append(radius)
                resample_ref_radii.append(ref_radius)

        self.segs_stenosis_key = [
            ["STENOSIS" if v > stenosis_threshold else "NONE" for v in branch_vals]
            for branch_vals in segs_stenosis_value
        ]

        self.segments = resample_segments
        self.radii = resample_radii
        self.ref_radii = resample_ref_radii
        self.segment_len = resample_lengths

        n_main = (len(self.segments) + 1) // 2
        self.ori_path = [
            p[0] for i in range(n_main) for p in self.ori_segments[2 * i]
        ]
        self.resample_path = [
            p[0] for i in range(n_main) for p in resample_segments[2 * i]
        ]
        self.resample_radiiseq = [
            r for i in range(n_main) for r in resample_radii[2 * i]
        ]


class Geometry_SynData(Geometry):

    def __init__(self, name: str, data=None):
        super().__init__(name)
        if data is not None:
            if isinstance(data, str):
                with open(data, "r") as f:
                    self.load_dict = json.load(f)
            else:
                self.load_dict = data
            self._set()

    def _set(self) -> None:
        segs = self.load_dict["segments"]
        try:
            segs.sort(key=lambda s: s["seg_id"])
        except KeyError:
            pass

        self.segments = []
        self.radii = []
        self.ref_radii = []
        self.links = []

        side_dir = 1

        for seg in segs:
            pos_3d = np.array(seg["pos"])
            if len(pos_3d) > 1:
                arc_len = np.concatenate(
                    ([0], np.cumsum(np.linalg.norm(np.diff(pos_3d, axis=0), axis=1)))
                )
            else:
                arc_len = np.array([0.0])

            seg_type = seg.get("type", "Main")
            if seg_type == "Side":
                y_vals = arc_len * side_dir
                side_dir *= -1
            else:
                y_vals = np.zeros_like(arc_len)

            self.segments.append([[s, y] for s, y in zip(arc_len, y_vals)])
            self.radii.append(seg["radius"])
            self.ref_radii.append(seg["ref_radius"])

            parent_id = seg["seg_id"]
            for child_id in seg.get("children", []):
                self.links.append([parent_id, child_id])

        self.labels = self.load_dict.get("stenosis_labels", [])
        self.conditions = self.load_dict.get("boundary_conditions", {})
        self.measure_site = self.load_dict.get("measure_site", {})
        self.scale = 1.0


def _closest_index(lst: list, x: float) -> int:
    return int(np.argmin([abs(v - x) for v in lst]))


def _has_duplicates(lst: list) -> bool:
    return len(lst) != len(set(lst))


def smooth_signal(signal, window_size: int) -> np.ndarray:
    out = []
    for i in range(len(signal)):
        lo = max(0, i - window_size // 2)
        hi = min(len(signal), i + window_size // 2 + 1)
        out.append(sum(signal[lo:hi]) / (hi - lo))
    return np.array(out)


def fit_reference_diameters(diameters) -> np.ndarray:
    diameters = np.array(diameters)
    filtered = smooth_signal(diameters, 25)
    x = np.arange(len(filtered))

    def line(x, a, b):
        return a * x + b

    try:
        popt, _ = curve_fit(line, x, filtered)
        fitted = line(x, *popt)
        mask = fitted > diameters
        filtered[mask] = fitted[mask]
        popt, _ = curve_fit(line, x, filtered)
        return line(x, *popt)
    except Exception:
        return filtered


def find_deepest_valley(diameters) -> tuple:
    diameters = np.array(diameters)
    min_idx = int(np.argmin(diameters))
    return max(0, min_idx - 5), min(len(diameters) - 1, min_idx + 5)
