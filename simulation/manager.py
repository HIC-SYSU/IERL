import os
import sys
import yaml
import numpy as np
import multiprocessing

from .geometry import Geometry_SynData
from .solver import SimulationSolver
from . import oneDSolver as _sv

__all__ = ["SimulationManager", "write_input_file"]

_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "simulation_config.yaml")


class SimulationManager:

    def __init__(
        self,
        tree: dict,
        save_path: str,
        dtype: str = "SynData",
        config_path: str = _DEFAULT_CONFIG,
    ):
        self.tree = tree
        self.dtype = dtype
        self.save_path = save_path
        self.config_path = config_path
        self.geometry: Geometry_SynData = None
        self.sim_solver: SimulationSolver = None
        self.pressure_list = None
        self.velocity_list = None
        self.FFR: float = None

        self._load_geometry()

    def _load_geometry(self) -> None:
        if self.dtype != "SynData":
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        self.geometry = Geometry_SynData(self.tree["case_id"], self.tree)
        self.geometry.set_segnum(seg_size=1.0)
        self.geometry.self_adaptive_segments()

    def setup_solver(self, pressure: float, velocity: float) -> None:
        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f)["simulation_parameters"]

        self.sim_solver = SimulationSolver(
            name=self.geometry.name,
            p=pressure,
            v=velocity,
            elenum=cfg["elenum"],
            outletbctype=cfg["outletbctype"],
            distributionlaw=cfg["distributionlaw"],
            geometry=self.geometry,
        )
        self.sim_solver.set_outlet_data(beta=cfg["beta"])
        self.sim_solver.set_BoundaryCondition(cfg["inlet_type"])
        self.sim_solver.set_material(cfg["density"], cfg["viscosity"], cfg["ref_pressure"])
        self.sim_solver.set_solver(
            cfg["time_step"], cfg["save_step"], cfg["max_step"],
            cfg["inlet_type"], cfg["convs"],
        )
        write_input_file(self.sim_solver, self.save_path, cfg["outputType"])

    def run(self) -> None:
        run_dir = os.path.dirname(self.save_path)
        filename = os.path.basename(self.save_path)
        proc = multiprocessing.Process(target=_run_solver, args=(run_dir, filename))
        proc.start()

    def run_blocking(self, timeout: float = 60.0, isolated: bool = True) -> None:
        run_dir = os.path.dirname(self.save_path)
        filename = os.path.basename(self.save_path)

        if isolated and multiprocessing.current_process().daemon:
            isolated = False

        if isolated:
            proc = multiprocessing.Process(
                target=_run_solver_inprocess, args=(run_dir, filename)
            )
            proc.start()
            proc.join(timeout=timeout)

            if proc.is_alive():
                proc.terminate()
                proc.join()
                raise RuntimeError(
                    f"Solver timed out after {timeout}s for '{self.save_path}'"
                )

            if proc.exitcode != 0:
                raise RuntimeError(
                    f"Solver crashed (exit code {proc.exitcode}) for '{self.save_path}'"
                )
        else:
            _run_solver_inprocess(run_dir, filename)

    def extract_results(self) -> None:
        base_dir = os.path.dirname(self.save_path)

        pressure_resampled = _read_pressure(base_dir)
        flow_resampled     = _read_flow(base_dir)

        self.pressure_list = _interp_to_original(pressure_resampled, self.geometry)
        self.flow_list     = _interp_to_original(flow_resampled,     self.geometry)

        site = self.geometry.measure_site
        inlet_p = self.pressure_list[0][0]
        measure_p = self.pressure_list[site["seg_id"]][site["local_idx"]]
        self.FFR  = measure_p / inlet_p
        self.ffr_list = [[p / inlet_p for p in branch] for branch in self.pressure_list]


def write_input_file(solver: SimulationSolver, path: str, output_type: str = "TEXT") -> None:
    lines = [
        "# ================================ \n"
        "# main MODEL - UNITS IN CGS \n"
        "# ================================ \n\n",
        "MODEL %s\n\n\n\n" % solver.name,
        "# ========== \n# NODE CARD \n# ========== \n\n",
    ]

    for nid, (x, y, z) in enumerate(solver.nodes):
        lines.append("NODE %d %f %f %f \n" % (nid, x, y, z))

    lines.append("\n# ========== \n# JOINT CARD \n# ========== \n\n")
    for j in solver.joints:
        lines.append(
            "JOINT %s %d %s %s \n" % (j["J_name"], j["J_node"],
                                       j["J_inlet"]["name"], j["J_outlet"]["name"])
        )
        lines.append("JOINTINLET %s %d %s\n" % (
            j["J_inlet"]["name"], j["J_inlet"]["num_seg"],
            " ".join(str(s) for s in j["J_inlet"]["seg_list"]),
        ))
        lines.append("JOINTOUTLET %s %d %s\n\n" % (
            j["J_outlet"]["name"], j["J_outlet"]["num_seg"],
            " ".join(str(s) for s in j["J_outlet"]["seg_list"]),
        ))

    lines.append("\n# ============= \n# SEGMENT CARD \n# ============= \n\n")
    for seg in solver.segments:
        lines.append(
            "SEGMENT %s %d %f %d %d %d %f %f %f %s %s %f %d %d %s %s \n" % (
                seg["name"], seg["seg_id"], seg["seg_len"], seg["total_ele"],
                seg["in_node"], seg["out_node"], seg["in_area"], seg["out_area"],
                seg["in_flow"], seg["material"], seg["loss_type"], seg["branch_angle"],
                seg["upstream"], seg["branch_seg_id"],
                seg["boundary_condition_type"], seg["data_table"],
            )
        )

    lines.append("\n")
    for tbl in solver.TableData:
        lines.append("DATATABLE %s LIST \n" % tbl["name"])
        for x, y in tbl["list_data"]:
            lines.append("%f %f \n" % (x, y))
        lines.append("ENDDATATABLE \n\n")

    lines.append("\n# ============== \n# MATERIAL CARD \n# ============== \n\n")
    for mat in solver.material:
        lines.append(
            "MATERIAL %s %s %f %f %f %f %f %f %f \n" % (
                mat["name"], mat["type"], mat["density"], mat["viscosity"],
                mat["ref_pressure"], mat["exponent"],
                mat["parameter_1"], mat["parameter_2"], mat["parameter_3"],
            )
        )

    lines.append("\n# ==================== \n# SOLVEROPTIONS CARD \n# ==================== \n\n")
    for slv in solver.solver:
        lines.append(
            "SOLVEROPTIONS %f %d %d %d %s %s %f %d %d \n" % (
                slv["time_step"], slv["save_step"], slv["max_step"],
                slv["quadrature_points"], slv["inlet_table"], slv["inlet_type"],
                slv["convs_torlerance"], slv["formulation_type"], slv["stabilization"],
            )
        )

    lines.append("\nOUTPUT %s" % output_type)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _run_solver(run_dir: str, filename: str) -> None:
    pass


def _run_solver_inprocess(run_dir: str, filename: str) -> None:
    orig_dir = os.getcwd()
    try:
        os.chdir(run_dir)
        _sv.resetGlobals()
        _sv.runFromFile(filename)
    finally:
        os.chdir(orig_dir)
        _sv.resetGlobals()


def _read_pressure(dir_path: str) -> list:
    files = os.listdir(dir_path)
    pressure_files = [f for f in files if "pressure.dat" in f]
    in_file = next(
        (os.path.splitext(f)[0] for f in files if f.endswith(".in")), ""
    )

    branch_seg_counts = _get_branch_seg_numbers(pressure_files)
    pressure_list = []

    for branch_idx, seg_count in enumerate(branch_seg_counts):
        branch_data = []
        for seg in range(int(seg_count) + 1):
            fname = f"{in_file}branch{branch_idx}_seg{seg}_pressure.dat"
            fpath = os.path.join(dir_path, fname)
            with open(fpath) as fh:
                lines = fh.readlines()
            branch_data.append(float(lines[0].split()[1]) / 10600)
        branch_data.append(float(lines[-1].split()[1]) / 10600)
        pressure_list.append(branch_data)

    return pressure_list


def _read_flow(dir_path: str) -> list:
    files = os.listdir(dir_path)
    flow_files = [f for f in files if "flow.dat" in f]
    in_file = next(
        (os.path.splitext(f)[0] for f in files if f.endswith(".in")), ""
    )

    branch_seg_counts = _get_branch_seg_numbers(flow_files)
    flow_list = []

    for branch_idx, seg_count in enumerate(branch_seg_counts):
        branch_data = []
        for seg in range(int(seg_count) + 1):
            fname = f"{in_file}branch{branch_idx}_seg{seg}_flow.dat"
            fpath = os.path.join(dir_path, fname)
            with open(fpath) as fh:
                lines = fh.readlines()
            branch_data.append(float(lines[0].split()[1]))
        branch_data.append(float(lines[-1].split()[1]))
        flow_list.append(branch_data)

    flow_list = [[f / 1000 for f in branch] for branch in flow_list]

    return flow_list


def _interp_to_original(values_resampled: list, geometry) -> list:
    result = []
    for i, vals in enumerate(values_resampled):
        seg_orig = geometry.ori_segments[i]
        arc_orig = [p[0] for p in seg_orig]

        if i % 2 == 0:
            sample_idx = geometry.sample_1d_indexs[i // 2]
            arc_sampled = [seg_orig[j][0] for j in sample_idx]
            interpolated = np.interp(arc_orig, arc_sampled, vals)
        else:
            interpolated = vals

        result.append(interpolated.tolist() if isinstance(interpolated, np.ndarray) else interpolated)

    return result


def _get_branch_seg_numbers(file_names: list) -> np.ndarray:
    branch_num = max(
        (int(f.split("branch")[-1].split("_")[0]) for f in file_names), default=0
    )
    seg_counts = np.zeros(branch_num + 1)
    for i in range(branch_num + 1):
        label = f"branch{i}_"
        branch_files = [f for f in file_names if label in f]
        if branch_files:
            seg_counts[i] = max(
                int(f.split("seg")[-1].split("_")[0]) for f in branch_files
            )
    return seg_counts
