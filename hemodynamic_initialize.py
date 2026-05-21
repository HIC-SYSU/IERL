import os
import sys
import glob
import json
import argparse
import tempfile
import shutil
import gc
import queue as _queue
import numpy as np
from multiprocessing import Pool, cpu_count

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROGRESS_PATH = os.path.join(SCRIPT_DIR, "progress.json")
DATA_DIR     = '/train'
TMP_BASE      = os.path.join(DATA_DIR, "tmp")

sys.path.insert(0, SCRIPT_DIR)

from simulation.manager import SimulationManager
from simulation.interface import SimulationInterface, _parse_bc


def load_done() -> tuple[set, set]:
    if os.path.exists(PROGRESS_PATH):
        try:
            with open(PROGRESS_PATH, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return set(data), set()
            return set(data.get("done", [])), set(data.get("failed", []))
        except Exception:
            print(f"[WARN] Progress file corrupted, regenerating: {PROGRESS_PATH}")
    return set(), set()


def mark_done(done_set: set, failed_set: set,
              stem: str | None, status: str = "done") -> None:
    if stem is not None:
        if status == "failed":
            failed_set.add(stem)
        else:
            done_set.add(stem)
    tmp = PROGRESS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(
            {"done": sorted(done_set), "failed": sorted(failed_set)},
            f, indent=2, ensure_ascii=False,
        )
    os.replace(tmp, PROGRESS_PATH)


def _cleanup_stem_tmp(stem: str) -> None:
    for name in os.listdir(TMP_BASE):
        if name.startswith(f"{stem}_"):
            path = os.path.join(TMP_BASE, name)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)


def compute_one(args: tuple) -> tuple:
    filepath, dst_dir = args
    ori_stem = os.path.splitext(os.path.basename(filepath))[0]
    out_name = f"{ori_stem}_hemo.json"
    out_path = os.path.join(dst_dir, out_name)
    tmp_dir  = None

    def _result(msg):
        return (ori_stem, msg)

    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        if "segments" not in data:
            return _result(f"[SKIP] {ori_stem} — no segments field")

        print(f"[INFO] Processing {ori_stem} ...")
        pressure_bc, velocity = _parse_bc(data)
        tmp_dir   = tempfile.mkdtemp(prefix=f"{ori_stem}_", dir=TMP_BASE)
        save_path = os.path.join(tmp_dir, f"{data.get('case_id', ori_stem)}.in")

        try:
            mgr = SimulationManager(data, save_path)
            mgr.setup_solver(pressure_bc, velocity)
            mgr.run_blocking()
            mgr.extract_results()

            measured_ffr  = float(mgr.FFR)
            pressure_tree = mgr.pressure_list
            flow_tree     = mgr.flow_list
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir = None

        del mgr
        gc.collect()

        if np.isnan(measured_ffr) or np.isinf(measured_ffr):
            raise ValueError(f"Invalid FFR value: {measured_ffr}")
        if not pressure_tree or not pressure_tree[0]:
            raise ValueError("Pressure result is empty")
        inlet_p = pressure_tree[0][0]
        if inlet_p <= 0 or np.isnan(inlet_p) or np.isinf(inlet_p):
            raise ValueError(f"Invalid inlet pressure: {inlet_p}")

        ffr_tree = [[p / inlet_p for p in branch] for branch in pressure_tree]

        sim = SimulationInterface(data)
        profile = {
            "measured_ffr": measured_ffr,
            "ffr":          ffr_tree,
            "pressure":     pressure_tree,
            "flow":         flow_tree,
        }
        main_profile = sim.extract_main_branch_profile(profile)
        del sim, profile

        x_main    = main_profile["x"].tolist()
        ffr_main  = main_profile["ffr"].tolist()
        pres_main = main_profile["pressure"].tolist()
        flow_main = main_profile["flow"].tolist()
        del main_profile

        if os.path.abspath(out_path) == os.path.abspath(filepath):
            raise ValueError(f"Output path is the same as source file, write refused: {out_path}")

        data["hemodynamics"] = {
            "measured_ffr":  measured_ffr,
            "ffr":           ffr_tree,
            "pressure":      pressure_tree,
            "flow":          flow_tree,
            "x_main":        x_main,
            "ffr_main":      ffr_main,
            "pressure_main": pres_main,
            "flow_main":     flow_main,
        }
        n_main = len(x_main)
        del pressure_tree, flow_tree, ffr_tree, x_main, ffr_main, pres_main, flow_main

        with open(out_path, "w") as f:
            json.dump(data, f)

        del data
        gc.collect()

        return _result(
            f"[OK  ] {ori_stem} → {out_name}"
            f"  FFR={measured_ffr:.3f}  main branch nodes={n_main}"
        )

    except Exception as e:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return _result(f"[FAIL] {ori_stem} — {str(e).splitlines()[0]}")


def run(data_dir: str, dst_dir: str, workers: int) -> None:
    dst_abs  = os.path.abspath(dst_dir)
    data_abs = os.path.abspath(data_dir)
    os.makedirs(dst_abs, exist_ok=True)

    os.makedirs(TMP_BASE, exist_ok=True)
    leftover = [d for d in os.listdir(TMP_BASE)
                if os.path.isdir(os.path.join(TMP_BASE, d))]
    if leftover:
        print(f"[INFO] Cleaning up {len(leftover)} leftover temp directories from last run...")
        for d in leftover:
            shutil.rmtree(os.path.join(TMP_BASE, d), ignore_errors=True)

    dst_prefix = dst_abs + os.sep
    all_json = [
        p for p in sorted(glob.glob(os.path.join(data_abs, "**", "*.json"), recursive=True))
        if not os.path.abspath(p).startswith(dst_prefix)
        and not p.endswith("_hemo.json")
        and os.path.abspath(p) != PROGRESS_PATH
    ]

    done_set, failed_set = load_done()

    existing_stems = {
        os.path.basename(p)[: -len("_hemo.json")]
        for p in glob.glob(os.path.join(dst_abs, "*_hemo.json"))
    }
    newly_recognized = existing_stems - done_set
    if newly_recognized:
        done_set |= newly_recognized
        mark_done(done_set, failed_set, None)
        print(f"  Identified {len(newly_recognized)} completed cases from existing output files, synced to progress file")

    skipped_set = done_set | failed_set
    tasks = [
        (os.path.abspath(p), dst_abs)
        for p in all_json
        if os.path.splitext(os.path.basename(p))[0] not in skipped_set
    ]

    print(f"Scanned {len(all_json)} source files")
    print(f"  Succeeded: {len(done_set)}  Failed (skipped): {len(failed_set)}  Pending: {len(tasks)}")
    print(f"Output dir: {dst_abs}  Parallel workers: {workers}")
    print()

    if not tasks:
        print("No files to process.")
        return

    mark_done(done_set, failed_set, None)

    done = err = skipped = 0

    result_q = _queue.Queue()
    pending = {os.path.splitext(os.path.basename(t[0]))[0] for t in tasks}

    with Pool(processes=workers, maxtasksperchild=1) as pool:
        for t in tasks:
            stem = os.path.splitext(os.path.basename(t[0]))[0]
            pool.apply_async(
                compute_one, (t,),
                callback=result_q.put,
                error_callback=lambda exc, s=stem: result_q.put(
                    (s, f"[FAIL] {s} — subprocess crashed: {type(exc).__name__}")
                ),
            )

        for i in range(1, len(tasks) + 1):
            try:
                stem_result, msg = result_q.get(timeout=600)
            except _queue.Empty:
                print(f"[WARN] Wait timeout, completed {i - 1}/{len(tasks)}")
                break

            pending.discard(stem_result)
            print(f"[{i}/{len(tasks)}] {msg}")
            if msg.startswith("[OK"):
                done += 1
                mark_done(done_set, failed_set, stem_result, "done")
            elif msg.startswith("[FAIL"):
                err += 1
                mark_done(done_set, failed_set, stem_result, "failed")
                _cleanup_stem_tmp(stem_result)
            else:
                skipped += 1
                mark_done(done_set, failed_set, stem_result, "done")

    if pending:
        print(f"\n[WARN] {len(pending)} tasks returned no result, marked as failed:")
        for stem in sorted(pending):
            print(f"  [LOST] {stem}")
            err += 1
            mark_done(done_set, failed_set, stem, "failed")
            _cleanup_stem_tmp(stem)

    leftover = [d for d in os.listdir(TMP_BASE)
                if os.path.isdir(os.path.join(TMP_BASE, d))]
    if leftover:
        print(f"[INFO] Cleaning up {len(leftover)} leftover temp directories from this run...")
        for d in leftover:
            shutil.rmtree(os.path.join(TMP_BASE, d), ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"Run complete  succeeded={done}  failed={err}  skipped={skipped}")
    print(f"Total succeeded={len(done_set)}  total failed={len(failed_set)}")
    print(f"Progress file: {PROGRESS_PATH}")
    print(f"During training, point data_dir to {dst_abs} to skip real-time CFD")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    parser = argparse.ArgumentParser(description="Initialize hemodynamics → *_hemo.json")
    parser.add_argument(
        "--data_dir",
        default=r"/train/synthetic_dataset_aug10x",
        help="Source JSON data directory",
    )
    parser.add_argument(
        "--dst_dir",
        default=r"/train/synthetic_dataset_aug10x_hemo",
        help="Output directory (where _hemo.json files are saved)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help=f"Number of parallel workers (default=CPU count, current={cpu_count()})",
    )
    args = parser.parse_args()
    run(args.data_dir, args.dst_dir, args.workers)
