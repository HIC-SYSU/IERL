import copy
from collections import OrderedDict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from simulation.interface import SimDivergenceError

_DEFAULT_STENT_DIAMETERS_MM = [2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
                                4.0, 4.25, 4.5, 4.75, 5.0]
_DEFAULT_STENT_LENGTHS_MM   = [8, 12, 16, 20, 24, 28, 32]


def apply_stent(x_mm: np.ndarray, area_mm2: np.ndarray,
                center_mm: float, length_mm: float, diam_mm: float) -> np.ndarray:
    stent_area = np.pi * (diam_mm / 2.0) ** 2
    a = center_mm - length_mm / 2.0
    b = center_mm + length_mm / 2.0
    mask = (x_mm >= a) & (x_mm <= b)
    new_area = area_mm2.copy()
    new_area[mask] = np.maximum(new_area[mask], stent_area)
    return new_area


class StentPlacementEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, dataset,
                 L:      int   = 100,
                 alpha1: float = 1.0,
                 alpha2: float = 0.5,
                 lam:    float = 1.0,
                 omega:  float = 0.1,
                 lam1:   float = 0.5,
                 lam2:   float = 0.5,
                 stent_diameters_mm=None,
                 stent_lengths_mm=None,
                 q_hyperemia_ml_s: float = 1.0,
                 p_inlet_mmhg:     float = 100.0,
                 meta_cache_size:  int | None = 64,
                 sim_interface=None):
        super().__init__()

        self.dataset          = dataset
        self.L                = L
        self.alpha1           = alpha1
        self.alpha2           = alpha2
        self.lam              = lam
        self.omega            = omega
        self.lam1             = lam1
        self.lam2             = lam2
        self.q_hyperemia      = q_hyperemia_ml_s
        self.p_inlet_mmhg     = p_inlet_mmhg
        self.meta_cache_size  = meta_cache_size
        self.sim_interface = sim_interface

        self.stent_diameters = np.array(
            stent_diameters_mm if stent_diameters_mm is not None
            else _DEFAULT_STENT_DIAMETERS_MM, dtype=np.float32)
        self.stent_lengths   = np.array(
            stent_lengths_mm if stent_lengths_mm is not None
            else _DEFAULT_STENT_LENGTHS_MM, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(L * 4,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.current: dict = {}
        self._meta_cache: OrderedDict[int, dict] = OrderedDict()

        self._rng = None
        self._epoch_indices: list = []
        self._cursor: int = 0

    @staticmethod
    def _resample(arr: np.ndarray, n: int) -> np.ndarray:
        if len(arr) == n:
            return arr.astype(np.float64)
        x_old = np.linspace(0.0, 1.0, len(arr))
        x_new = np.linspace(0.0, 1.0, n)
        return np.interp(x_new, x_old, arr)

    def _build_meta(self, sample: dict) -> dict:
        x_grid = np.asarray(sample["x_grid"], dtype=np.float64)
        x_raw  = np.asarray(sample.get("x_raw", sample["x"]), dtype=np.float64)
        d_raw  = np.asarray(sample.get("d_raw", sample["d"]), dtype=np.float64)

        d_on_grid = np.interp(x_grid, x_raw, d_raw)
        area      = np.pi * (d_on_grid / 2.0) ** 2

        tree_dict    = sample.get("tree_dict", {})
        measured_ffr = 1.0

        if ("pressure" in sample and "flow" in sample and "ffr" in sample
                and sample["pressure"] is not None
                and sample["flow"]     is not None
                and sample["ffr"]      is not None):
            x_src    = np.asarray(sample.get("x_precomp", x_raw), dtype=np.float64)
            ffr      = np.interp(x_grid, x_src, np.asarray(sample["ffr"],      dtype=np.float64))
            pressure = np.interp(x_grid, x_src, np.asarray(sample["pressure"], dtype=np.float64))
            flow     = np.interp(x_grid, x_src, np.asarray(sample["flow"],     dtype=np.float64))
            measured_ffr = sample.get("measured_ffr", float(ffr[-1]))

        elif self.sim_interface is not None and "tree_dict" in sample:
            sim  = self.sim_interface(sample["tree_dict"])
            prof = sim.compute_hemodynamic_profile()
            measured_ffr = prof.get("measured_ffr", 1.0)
            x_cfd    = np.asarray(prof.get("x", x_raw), dtype=np.float64)
            ffr      = np.interp(x_grid, x_cfd, np.asarray(prof["ffr"],      dtype=np.float64))
            pressure = np.interp(x_grid, x_cfd, np.asarray(prof["pressure"], dtype=np.float64))
            flow     = np.interp(x_grid, x_cfd, np.asarray(prof["flow"],     dtype=np.float64))

        else:
            raise RuntimeError(
                f"[StentPlacementEnv] No hemodynamic data for sample "
                f"'{sample.get('patient', '?')}': provide pre-computed arrays "
                f"or a sim_interface."
            )

        return {
            "x":            x_grid,
            "x_raw":        x_raw,
            "x_grid":       x_grid,
            "area":         area,
            "measured_ffr": measured_ffr,
            "ffr":          ffr,
            "pressure":     pressure,
            "flow":         flow,
            "total_length":    float(x_grid[-1]),
            "measure_site_mm": float(sample.get("measure_site_mm", x_grid[-1])),
            "ref":             sample.get("ref_stent", None),
            "tree_dict":       tree_dict,
            "patient":         sample.get("patient", ""),
        }

    def _make_obs(self, meta: dict) -> np.ndarray:
        x        = meta["x"]
        area     = meta["area"]
        pressure = meta["pressure"]
        flow     = meta["flow"]

        span   = max(float(x[-1] - x[0]), 1e-6)
        x_norm = ((x - x[0]) / span).clip(0.0, 1.0)

        proximal_area = max(float(area[0]), 1e-6)
        area_norm = (area / proximal_area).clip(0.0, 2.0)

        inlet_p = max(float(pressure[0]), 1e-6)
        inlet_q = max(float(np.abs(flow[0])), 1e-6)
        p_norm  = (pressure / inlet_p).clip(0.0, 1.0)
        f_norm  = (np.abs(flow) / inlet_q).clip(0.0, 1.0)


        obs = np.stack([x_norm, area_norm, p_norm, f_norm], axis=-1).flatten()
        return obs.astype(np.float32)

    @staticmethod
    def _delta_ffr(ffr_pre: float, ffr_post: float) -> float:
        return (ffr_post - ffr_pre) / max(1.0 - ffr_pre, 1e-6)

    @staticmethod
    def _smoothness(ffr: np.ndarray) -> float:
        n = len(ffr)
        if n < 3:
            return 0.0
        s = sum(abs(ffr[i] - 2.0 * ffr[i - 1] + ffr[i - 2]) for i in range(2, n))
        return -s / (n - 2)

    def _anatomical_penalty(self,
                            x_mm: np.ndarray, area_post: np.ndarray,
                            center_pred: float, len_pred: float,
                            ref) -> float:
        if ref is None:
            return 0.0

        ref_pos, ref_len, ref_diam = float(ref[0]), float(ref[1]), float(ref[2])

        a_ref  = ref_pos   - ref_len  / 2.0
        b_ref  = ref_pos   + ref_len  / 2.0
        a_pred = center_pred - len_pred / 2.0
        b_pred = center_pred + len_pred / 2.0

        intersect = max(0.0, min(b_ref, b_pred) - max(a_ref, a_pred))
        c_region  = 1.0 - intersect / max(b_ref - a_ref, 1e-6)

        mask = (x_mm >= a_pred) & (x_mm <= b_pred)
        if mask.sum() > 0:
            d_pred_mm = 2.0 * np.sqrt(area_post[mask].mean() / np.pi)
            c_size    = abs(ref_diam - d_pred_mm) / max(ref_diam, 1e-6)
        else:
            c_size = 1.0

        return self.lam1 * c_region + self.lam2 * c_size

    def _divergence_reward(self, ffr_pre: float) -> float:
        delta_ffr_worst = -ffr_pre / max(1.0 - ffr_pre, 1e-6)
        F_worst = self.lam * delta_ffr_worst
        return float(np.tanh(self.alpha1 * F_worst - self.alpha2 * 1.0))

    def reset(self, *, seed=None, options=None, idx: int = None):
        if idx is None:
            super().reset(seed=seed)
            if self._rng is None:
                self._rng = np.random.default_rng(seed)
            if self._cursor >= len(self._epoch_indices):
                self._epoch_indices = self._rng.permutation(len(self.dataset)).tolist()
                self._cursor = 0
            idx = self._epoch_indices[self._cursor]
            self._cursor += 1

        if self.meta_cache_size is not None and self.meta_cache_size <= 0:
            meta = self._build_meta(self.dataset[idx])
        else:
            meta = self._meta_cache.pop(idx, None)
            if meta is None:
                meta = self._build_meta(self.dataset[idx])
            self._meta_cache[idx] = meta
            if self.meta_cache_size is not None:
                while len(self._meta_cache) > self.meta_cache_size:
                    self._meta_cache.popitem(last=False)

        self.current = copy.deepcopy(meta)
        return self._make_obs(self.current), {}

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        meta   = self.current
        x      = meta["x"]

        total_len    = meta["total_length"]
        center_mm    = float(action[0]) * total_len
        len_idx      = int(round(float(action[1]) * (len(self.stent_lengths)   - 1)))
        diam_idx     = int(round(float(action[2]) * (len(self.stent_diameters) - 1)))
        stent_len_mm = float(self.stent_lengths[len_idx])
        stent_d_mm   = float(self.stent_diameters[diam_idx])

        x_grid  = meta["x_grid"]
        ffr_pre = float(meta["measured_ffr"])

        try:
            sim       = self.sim_interface(meta["tree_dict"])
            post_tree = sim.apply_stent_to_tree(center_mm, stent_len_mm, stent_d_mm)
            post_sim  = self.sim_interface(post_tree)
            prof      = post_sim.compute_hemodynamic_profile()

            x_cfd_post    = np.asarray(prof.get("x", meta["x_raw"]), dtype=np.float64)
            ffr_post      = np.interp(x_grid, x_cfd_post, np.asarray(prof["ffr"],      dtype=np.float64))
            pressure_post = np.interp(x_grid, x_cfd_post, np.asarray(prof["pressure"], dtype=np.float64))
            flow_post     = np.interp(x_grid, x_cfd_post, np.asarray(prof["flow"],     dtype=np.float64))
            area_post     = apply_stent(x_grid, meta["area"], center_mm, stent_len_mm, stent_d_mm)

            measured_ffr_post = float(prof.get("measured_ffr", 1.0))

            delta_ffr = self._delta_ffr(ffr_pre, measured_ffr_post)
            f_smooth  = self._smoothness(ffr_post)
            F         = self.lam * delta_ffr + self.omega * f_smooth

            P = self._anatomical_penalty(x, area_post,
                                         center_mm, stent_len_mm,
                                         meta["ref"])

            reward = float(np.tanh(self.alpha1 * F - self.alpha2 * P))

            meta["area"]      = area_post
            meta["ffr"]       = ffr_post
            meta["pressure"]  = pressure_post
            meta["flow"]      = flow_post
            meta["tree_dict"] = post_tree

            obs = self._make_obs(meta)
            info = {
                "pred_mm":    np.array([center_mm, stent_len_mm, stent_d_mm], dtype=np.float32),
                "ref_mm":     np.array(meta["ref"], dtype=np.float32)
                              if meta["ref"] is not None else np.zeros(3, dtype=np.float32),
                "delta_ffr":  delta_ffr,
                "ffr_pre":    ffr_pre,
                "ffr_post":   float(measured_ffr_post),
                "sim_failed": False,
                "diverged":   False,
            }

        except (SimDivergenceError, Exception) as exc:
            diverged = isinstance(exc, SimDivergenceError)
            print(f"[StentPlacementEnv] {'Divergence' if diverged else 'Sim failed'} "
                  f"for '{meta['patient']}' "
                  f"action=({center_mm:.1f}mm, {stent_len_mm:.0f}mm, {stent_d_mm:.2f}mm): {exc}")
            reward = self._divergence_reward(ffr_pre)
            obs    = self._make_obs(meta)
            info   = {
                "pred_mm":    np.array([center_mm, stent_len_mm, stent_d_mm], dtype=np.float32),
                "ref_mm":     np.array(meta["ref"], dtype=np.float32)
                              if meta["ref"] is not None else np.zeros(3, dtype=np.float32),
                "delta_ffr":  -ffr_pre / max(1.0 - ffr_pre, 1e-6),
                "ffr_pre":    ffr_pre,
                "ffr_post":   0.0,
                "sim_failed": True,
                "diverged":   diverged,
            }

        return obs, reward, True, False, info
