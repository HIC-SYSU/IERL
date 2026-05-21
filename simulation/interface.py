import copy
import multiprocessing as mp
import os
import queue
import shutil
import tempfile
from typing import Optional

import numpy as np

from .manager import SimulationManager, _get_branch_seg_numbers

__all__ = ["SimulationInterface", "SimDivergenceError"]


class SimDivergenceError(RuntimeError):
    pass


def _solver_worker(queue, tree_dict, config_path):
    try:
        profile = _compute_profile_blocking(tree_dict, config_path)
        queue.put(("ok", profile))
    except Exception as exc:
        queue.put(("err", str(exc)))

_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "simulation_config.yaml")

_DYNE_TO_MMHG = 1.0 / 1333.22

_SCRATCH = os.environ.get("SCRATCH")
_TMP_BASE = os.path.join(_SCRATCH, "tmp") if _SCRATCH else tempfile.gettempdir()
os.makedirs(_TMP_BASE, exist_ok=True)


def _default_start_method() -> str:
    env_override = os.environ.get("AVCIP_SIM_START_METHOD")
    if env_override:
        return env_override
    return "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"


def _compute_profile_blocking(tree_dict: dict, config_path: str) -> dict:
    pressure_bc, velocity = _parse_bc(tree_dict)
    tmp_dir = tempfile.mkdtemp(prefix="avcip_sim_", dir=_TMP_BASE)
    save_path = os.path.join(tmp_dir, f"{tree_dict.get('case_id', 'tree')}.in")
    try:
        mgr = SimulationManager(tree_dict, save_path, config_path=config_path)
        mgr.setup_solver(pressure_bc, velocity)
        mgr.run_blocking(isolated=False)
        mgr.extract_results()
        return {
            "measured_ffr": mgr.FFR,
            "ffr": mgr.ffr_list,
            "pressure": mgr.pressure_list,
            "flow": mgr.flow_list,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


class SimulationInterface:

    def __init__(self, tree_dict, config_path: Optional[str] = None, start_method: Optional[str] = None):
        self.config_path = config_path or _DEFAULT_CONFIG
        self.start_method = start_method or _default_start_method()
        self.tree_dict = tree_dict

    def compute_hemodynamic_profile(
        self,
        L: Optional[int] = None,
        isolated: Optional[bool] = None,
    ) -> dict:
        if isolated is None:
            isolated = not mp.current_process().daemon

        if isolated:
            ctx = mp.get_context(self.start_method)
            q = ctx.Queue()
            p = ctx.Process(target=_solver_worker, args=(q, self.tree_dict, self.config_path))
            p.start()
            p.join(timeout=600)

            crashed = (p.exitcode != 0) or (p.exitcode is None)
            if p.is_alive():
                p.kill()
                p.join()

            if crashed:
                raise SimDivergenceError(
                    f"C++ solver crashed (exitcode={p.exitcode}) "
                    f"for case '{self.tree_dict.get('case_id', '?')}'"
                )

            try:
                status, *result = q.get(timeout=5.0)
            except queue.Empty as exc:
                raise SimDivergenceError(
                    f"C++ solver returned no results for case '{self.tree_dict.get('case_id', '?')}'"
                ) from exc
            if status == "err":
                raise SimDivergenceError(result[0])

            profile = result[0]
        else:
            try:
                profile = _compute_profile_blocking(self.tree_dict, self.config_path)
            except Exception as exc:
                raise SimDivergenceError(str(exc)) from exc

        return self.extract_main_branch_profile(profile)

    def extract_main_branch_profile(self, profile: dict) -> dict:
        segs       = self.tree_dict["segments"]
        main_index = [i for i, seg in enumerate(segs) if seg.get("type") in ("Root", "Main") or i % 2 == 0]

        offsets: dict = {}
        for i, seg in enumerate(segs):
            if i == 0:
                offsets[i] = 0.0
            else:
                parent_i = next((j for j, s in enumerate(segs)
                                 if i in s.get("children", [])), None)
                if parent_i is not None and parent_i in offsets:
                    offsets[i] = offsets[parent_i] + segs[parent_i].get("length", 0.0)
                else:
                    offsets[i] = 0.0

        def concat_with_x(key):
            arrays_x, arrays_v = [], []
            for i in main_index:
                seg = segs[i]
                n   = len(profile[key][i])
                arc0 = offsets.get(i, 0.0)
                arc1 = arc0 + seg.get("length", 0.0)
                arrays_x.append(np.linspace(arc0, arc1, n))
                arrays_v.append(np.asarray(profile[key][i], dtype=np.float64))
            return np.concatenate(arrays_x), np.concatenate(arrays_v)

        x_cfd, ffr_cfd      = concat_with_x("ffr")
        _,     pressure_cfd = concat_with_x("pressure")
        _,     flow_cfd     = concat_with_x("flow")

        return {
            "measured_ffr": profile.get("measured_ffr", 1.0),
            "x":            x_cfd.astype(np.float32),
            "ffr":          ffr_cfd.astype(np.float32),
            "pressure":     pressure_cfd.astype(np.float32),
            "flow":         flow_cfd.astype(np.float32),
        }

    def apply_stent_to_tree(
        self,
        center_mm: float,
        length_mm: float,
        diam_mm: float,
    ) -> dict:
        stent_r = diam_mm / 2.0
        s_start = max(0.0, center_mm - length_mm / 2.0)
        s_end   = center_mm + length_mm / 2.0

        td   = copy.deepcopy(self.tree_dict)
        segs = {s["seg_id"]: s for s in td["segments"]}

        root_id = next(sid for sid, s in segs.items() if s["parent"] is None)
        offsets: dict = {}
        queue = [(root_id, 0.0)]
        while queue:
            sid, arc = queue.pop(0)
            offsets[sid] = arc
            for cid in segs[sid]["children"]:
                queue.append((cid, arc + segs[sid]["length"]))

        for sid, seg in segs.items():
            arc0 = offsets[sid]
            arc1 = arc0 + seg["length"]
            if arc1 < s_start or arc0 > s_end:
                continue
            pts = np.linspace(arc0, arc1, len(seg["radius"]))
            for i, p in enumerate(pts):
                if s_start <= p <= s_end:
                    seg["radius"][i] = max(stent_r, seg["radius"][i])

        return td


def _parse_bc(tree_dict: dict):
    bc = tree_dict.get("boundary_conditions", {})

    if "sampled_pressure" in bc:
        pressure_bc = float(bc["sampled_pressure"])
    else:
        pressure_bc = float(bc.get("pressure", 100))

    if "sampled_velocity" in bc:
        velocity = float(bc["sampled_velocity"])
    else:
        vel_range = bc.get("velocity_range")
        if isinstance(vel_range, (list, tuple)) and len(vel_range) == 2:
            velocity = float(vel_range[0] + vel_range[1]) / 2.0
        else:
            velocity = float(bc.get("velocity", 0.3))

    return pressure_bc, velocity


def _resample1d(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) == n:
        return arr
    x_old = np.linspace(0.0, 1.0, len(arr))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, arr).astype(arr.dtype)
