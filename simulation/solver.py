import numpy as np

__all__ = ["SimulationSolver"]


class SimulationSolver:
    def __init__(
        self,
        name: str,
        p: float,
        v: float,
        elenum: int = 8,
        outletbctype: str = "RESISTANCE",
        distributionlaw: str = "HK_main",
        geometry=None,
    ):
        self.name = name
        self.p_in = p * 1333.2
        self.v = v

        self.nodes: list = []
        self.joints: list = []
        self.segments: list = []
        self.TableData: list = []
        self.material: list = []
        self.solver: list = []

        self.branch_info: dict = {}
        self.branch_links: list = []
        self.Q_distribution: list = []
        self.Q_in: float = 0.0
        self.outlet_segs: list = []
        self.outlet_data: list = []
        self.ref_radii: list = []

        self.outletbctype = outletbctype
        self.distributionlaw = distributionlaw
        self.elenum = elenum

        if geometry is not None:
            self.set_geometry(geometry)

    def set_geometry(self, geometry) -> None:
        cur_node_id = cur_seg_id = cur_joint_id = 0
        branch_info: dict = {}
        nodes, segments, joints, outlet_segs, ref_radii = [], [], [], [], []

        for branch_id, (seg, radius, ref_radius, seg_len, sten_key) in enumerate(zip(
            geometry.segments,
            geometry.radii,
            geometry.ref_radii,
            geometry.segment_len,
            geometry.segs_stenosis_key,
        )):
            radius_arr = np.asarray(radius) * 0.1
            ref_radius_arr = np.asarray(ref_radius) * 0.1
            segment_arr = np.asarray(seg) * 0.1
            seg_len_arr = np.asarray(seg_len) * 0.1

            num_node = len(seg)
            node_start = cur_node_id
            node_end = cur_node_id + num_node - 1
            seg_start = cur_seg_id

            for point, rr in zip(segment_arr, ref_radius_arr):
                nodes.append((point[0], point[1], 0.0))
                ref_radii.append(rr)
                cur_node_id += 1

            for local in range(num_node - 1):
                in_node = node_start + local
                out_node = in_node + 1
                length = seg_len_arr[local] if local < len(seg_len_arr) else 0.0
                in_area = radius_arr[local] ** 2 * np.pi
                out_area = radius_arr[local + 1] ** 2 * np.pi
                loss = sten_key[local] if local < len(sten_key) else "NONE"
                if local == 0:
                    loss = "NONE"

                segments.append({
                    "material": "MAT1",
                    "total_ele": self.elenum,
                    "branch_angle": 0.0,
                    "upstream": max(cur_seg_id - 1, 0),
                    "loss_type": loss,
                    "branch_seg_id": 0,
                    "boundary_condition_type": "NOBOUND",
                    "data_table": "NONE",
                    "name": "branch%d_seg%d" % (branch_id, local),
                    "seg_id": cur_seg_id,
                    "in_node": in_node,
                    "out_node": out_node,
                    "seg_len": length,
                    "in_flow": 0.0,
                    "in_area": in_area,
                    "out_area": out_area,
                })
                cur_seg_id += 1

            for idx, joint_node in enumerate(range(node_start + 1, node_end)):
                joints.append({
                    "J_name": "J%d" % cur_joint_id,
                    "J_node": joint_node,
                    "J_inlet": {"name": "INT%d" % cur_joint_id, "num_seg": 1, "seg_list": [seg_start + idx]},
                    "J_outlet": {"name": "OUT%d" % cur_joint_id, "num_seg": 1, "seg_list": [seg_start + idx + 1]},
                })
                cur_joint_id += 1

            branch_info[branch_id] = {
                "branch_id": branch_id,
                "node_start": node_start,
                "node_end": node_end,
                "seg_start": seg_start,
                "seg_end": cur_seg_id - 1,
            }

        seg_links = np.asarray(geometry.links)
        if len(seg_links) > 0:
            for branch_id, info in branch_info.items():
                children = seg_links[seg_links[:, 0] == branch_id][:, 1]
                if len(children) > 0:
                    seg_list = [branch_info[c]["seg_start"] for c in children]
                    joints.append({
                        "J_name": "J%d" % cur_joint_id,
                        "J_node": info["node_end"],
                        "J_inlet": {"name": "INT%d" % cur_joint_id, "num_seg": 1, "seg_list": [info["seg_end"]]},
                        "J_outlet": {"name": "OUT%d" % cur_joint_id, "num_seg": len(seg_list), "seg_list": seg_list},
                    })
                    cur_joint_id += 1
                else:
                    outlet_segs.append(info["seg_end"])
                    segments[info["seg_end"]]["boundary_condition_type"] = self.outletbctype
        else:
            end = branch_info[0]["seg_end"]
            segments[end]["boundary_condition_type"] = self.outletbctype
            outlet_segs.append(end)

        self.Q_in = segments[0]["in_area"] * self.v * 100
        self.nodes = nodes
        self.segments = segments
        self.joints = joints
        self.outlet_segs = outlet_segs
        self.branch_info = branch_info
        self.ref_radii = ref_radii
        self.branch_links = geometry.links

    def set_outlet_data(self, beta: float) -> None:
        self.set_Q_distribution(self.Q_in, beta)
        vals = []
        if self.outletbctype == "RESISTANCE":
            for q in self.Q_distribution:
                seg_id = self.branch_info[q["branch_id"]]["seg_end"]
                if seg_id in self.outlet_segs:
                    vals.append({"seg_id": seg_id, "RESISTANCE": self.p_in / q["Q"]})
        elif self.outletbctype == "FLOW":
            for q in self.Q_distribution:
                seg_id = self.branch_info[q["branch_id"]]["seg_end"]
                if seg_id in self.outlet_segs:
                    vals.append({"seg_id": seg_id, "FLOW": q["Q"]})
        else:
            raise ValueError(f"Unsupported outletbctype: {self.outletbctype}")
        self.outlet_data = vals

    def set_Q_distribution(self, Q0: float, beta: float) -> None:
        if self.distributionlaw == "HK_main":
            self._flow_distribution_HK(Q0, beta)
        elif self.distributionlaw == "Murray":
            self._flow_distribution_Murray(Q0, beta)
        else:
            raise ValueError(f"Unsupported distributionlaw: {self.distributionlaw}")

    def _flow_distribution_HK(self, Q0: float, beta: float) -> None:
        dist = [{"branch_id": 0, "Q": Q0}]
        for main_id in range(2, len(self.branch_info), 2):
            up = self.branch_info[main_id - 2]
            dn = self.branch_info[main_id]
            Q_up = dist[main_id - 2]["Q"]
            Q_dn = Q_up * (self.ref_radii[dn["node_start"]] / self.ref_radii[up["node_end"]]) ** beta
            dist.append({"branch_id": main_id - 1, "Q": Q_up - Q_dn})
            dist.append({"branch_id": main_id, "Q": Q_dn})
        self.Q_distribution = dist

    def _flow_distribution_Murray(self, Q0: float, beta: float) -> None:
        dist = [{"branch_id": 0, "Q": Q0}]
        outlet_ids = []
        for seg_id in self.outlet_segs:
            for bid, info in self.branch_info.items():
                if info["seg_end"] == seg_id:
                    outlet_ids.append(bid)
                    break
        total_pote = sum(
            self.ref_radii[self.branch_info[bid]["node_end"]] ** beta
            for bid in outlet_ids
        )
        for bid in outlet_ids:
            r = self.ref_radii[self.branch_info[bid]["node_end"]]
            dist.append({"branch_id": bid, "Q": Q0 * r ** beta / total_pote})
        for bid in range(len(self.branch_info) - 1, 1, -1):
            if bid not in outlet_ids:
                children = [lnk[1] for lnk in self.branch_links if lnk[0] == bid]
                total = sum(item["Q"] for item in dist if item["branch_id"] in children)
                dist.append({"branch_id": bid, "Q": total})
        dist.sort(key=lambda x: x["branch_id"])
        self.Q_distribution = dist

    def set_BoundaryCondition(self, inlet_type: str) -> None:
        for num, data in enumerate(self.outlet_data):
            table_name = "RESTABLE%d" % num
            if self.outletbctype == "RESISTANCE":
                table_data = [(0.0, data["RESISTANCE"])]
            else:
                table_data = [(0.0, data["FLOW"])]
            self.TableData.append({"name": table_name, "list_data": table_data})
            self.segments[data["seg_id"]]["data_table"] = table_name

        if inlet_type == "FLOW":
            self.TableData.append({"name": "INLETTABLE", "list_data": [(0.0, self.Q_in), (1.0, self.Q_in)]})
        elif inlet_type == "PRESSURE_WAVE":
            self.TableData.append({"name": "INLETTABLE", "list_data": [(0.0, self.p_in), (1.0, self.p_in)]})
        else:
            raise ValueError(f"Unsupported inlet_type: {inlet_type}")

    def set_material(self, density: float, viscosity: float, ref_pressure: float) -> None:
        self.material.append({
            "name": "MAT1", "type": "LINEAR",
            "density": density, "viscosity": viscosity,
            "ref_pressure": ref_pressure,
            "exponent": 1,
            "parameter_1": 10_000_000.0,
            "parameter_2": 0.0,
            "parameter_3": 0.0,
        })

    def set_solver(
        self,
        time_step: float,
        save_step: int,
        max_step: int,
        inlet_type: str,
        convs: float,
    ) -> None:
        self.solver.append({
            "time_step": time_step,
            "save_step": save_step,
            "max_step": max_step,
            "quadrature_points": 2,
            "inlet_table": "INLETTABLE",
            "inlet_type": inlet_type,
            "convs_torlerance": convs,
            "formulation_type": 1,
            "stabilization": 1,
        })
