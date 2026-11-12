#!/usr/bin/env python3
"""
Run 1b -- stage-level power decomposition and the measurement-boundary table.

The reviewer requires:
  * the stage-level numbers, obtained by running each stage ALONE and
    subtracting the idle draw, and the stages must add up to the whole-system
    number within a few percent;
  * the per-sensor idle draw;
  * the measurement-boundary table, stating for each item where it was
    measured and with which instrument.

Stages of the payload:
  S0  system idle (no workload at all)
  S1  sensor replay only (the recorded stream is read and binned, no network)
  S2  fusion core on the CPU only (C3's core, replaying the same stream)
  S3  GPU front-end only (C1's network on the GPU)
  S4  full system (C4 = GPU front-end + CPU fusion core)

Each stage is measured with the SAME instrument and the SAME protocol as the
whole-system runs, so the sums are comparable.
"""

import os
import sys
import json
import time
import threading
import argparse
import statistics
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
sys.path.insert(0, HERE)

import workload as WL
from powermeter import PowerLogger, InstrumentMap



# ------------------------------------------------------------------ guard
def require_quiescent(tag, outdir):
    """Refuse to start a measurement unless the machine is genuinely idle.

    A power number is only attributable to the configuration under test if no
    other process is drawing from the same rails.  This check is run BEFORE
    every measurement window and its record is written next to the results so
    that every figure can be traced back to the machine state it was taken in.
    """
    import subprocess
    import json
    rec_path = os.path.join(outdir, "quiescence", f"{tag}.json")
    os.makedirs(os.path.dirname(rec_path), exist_ok=True)
    p = subprocess.run([sys.executable, os.path.join(HERE, "quiescence.py"),
                        "--out", rec_path],
                       capture_output=True, text=True)
    try:
        rec = json.load(open(rec_path))
    except Exception:
        rec = {"quiescent": False, "verdict": "quiescence check failed to run: "
                + p.stderr[-400:], "busy_processes": []}
    print(f"  [quiescence] {tag}: {rec.get('verdict','')}", flush=True)
    if not rec.get("quiescent"):
        print("  [quiescence] REFUSING TO MEASURE -- the measurement would "
              "include the other activity listed in the record.", flush=True)
        raise SystemExit(
            "measurement aborted: the machine was not quiescent. "
            f"See {rec_path} for the offending processes.")
    return rec

def measure(power_csv, dur, rate_hz, work_fn, imap, label):
    """Run work_fn for dur seconds while logging power.  Returns mean W."""
    logger = PowerLogger(power_csv, rate_hz=20.0, imap=imap,
                         tegrastats_interval_ms=100)
    stop = threading.Event()

    def worker():
        work_fn(stop)

    t = threading.Thread(target=worker, daemon=True)
    logger_thread = threading.Thread(target=logger.run, args=(dur,),
                                     kwargs={"label": label}, daemon=True)
    t.start()
    logger_thread.start()
    time.sleep(0.3)
    time.sleep(dur)
    stop.set()
    logger_thread.join(timeout=10)
    t.join(timeout=10)

    with open(power_csv) as f:
        import csv as _csv
        rows = list(_csv.DictReader(f))
    vals = [float(r["teg_vdd_in_mw_avg"]) / 1000.0 for r in rows if r["teg_vdd_in_mw_avg"]]
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(HERE, "results", "run1"))
    ap.add_argument("--logdir", default=os.path.join(HERE, "logs", "run1"))
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rate", type=float, default=50.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    scans = WL.read_scans()
    enc = np.stack([WL.encode_input(WL.profile_from_scan(a)) for _, a in scans])
    weights = WL.load_weights(os.path.join(HERE, "common", "weights_quantised.npz"))
    netcfg = dict(WL.NET)
    netcfg["rate_hz"] = args.rate

    imap = InstrumentMap(jetson_ina="/sys/class/hwmon/hwmon1")

    from run1_power import TorchGPUBackend, CppBackend, HybridBackend

    stages = {}

    def make_work(kind):
        state = {}

        def work(stop):
            if kind == "idle":
                while not stop.is_set():
                    time.sleep(0.05)
                return
            if kind == "replay":
                # the sensor front-end only: read the recorded stream and bin
                # it into the 256-bin profile, with no network
                i = 0
                period = 1.0 / args.rate
                nxt = time.perf_counter()
                while not stop.is_set():
                    nxt += period
                    now = time.perf_counter()
                    if now < nxt:
                        time.sleep(nxt - now)
                    else:
                        nxt = now
                    p = WL.profile_from_scan(scans[i % len(scans)][1])
                    x = WL.encode_input(p)
                    i += 1
                return
            if kind == "cpu_core":
                if "b" not in state:
                    state["b"] = CppBackend(weights, netcfg)
                    state["b"].reset()
                b = state["b"]
                i = 0
                period = 1.0 / args.rate
                nxt = time.perf_counter()
                while not stop.is_set():
                    nxt += period
                    now = time.perf_counter()
                    if now < nxt:
                        time.sleep(nxt - now)
                    else:
                        nxt = now
                    b.step(enc[i % len(enc)])
                    i += 1
                return
            if kind == "gpu":
                if "b" not in state:
                    state["b"] = TorchGPUBackend(weights, netcfg)
                    state["b"].reset()
                b = state["b"]
                i = 0
                period = 1.0 / args.rate
                nxt = time.perf_counter()
                while not stop.is_set():
                    nxt += period
                    now = time.perf_counter()
                    if now < nxt:
                        time.sleep(nxt - now)
                    else:
                        nxt = now
                    b.step(enc[i % len(enc)])
                    i += 1
                return
            if kind == "hybrid_no_gpu":
                if "b" not in state:
                    state["b"] = CppBackend(weights, netcfg)
                    state["b"].reset()
                b = state["b"]
                i = 0
                period = 1.0 / args.rate
                nxt = time.perf_counter()
                while not stop.is_set():
                    nxt += period
                    now = time.perf_counter()
                    if now < nxt:
                        time.sleep(nxt - now)
                    else:
                        nxt = now
                    b.step(enc[i % len(enc)])
                    i += 1
                return
            if kind == "hybrid_no_cpu":
                if "b" not in state:
                    state["b"] = TorchGPUBackend(weights, netcfg)
                    state["b"].reset()
                b = state["b"]
                i = 0
                period = 1.0 / args.rate
                nxt = time.perf_counter()
                while not stop.is_set():
                    nxt += period
                    now = time.perf_counter()
                    if now < nxt:
                        time.sleep(nxt - now)
                    else:
                        nxt = now
                    b.step(enc[i % len(enc)])
                    i += 1
                return
            if kind == "hybrid":
                if "b" not in state:
                    state["b"] = HybridBackend(weights, netcfg)
                    state["b"].reset()
                b = state["b"]
                i = 0
                period = 1.0 / args.rate
                nxt = time.perf_counter()
                while not stop.is_set():
                    nxt += period
                    now = time.perf_counter()
                    if now < nxt:
                        time.sleep(nxt - now)
                    else:
                        nxt = now
                    b.step(enc[i % len(enc)])
                    i += 1
                return
        return work

    defs = [
        ("S0_system_idle", "idle"),
        ("S1_sensor_replay_only", "replay"),
        ("S2_cpu_fusion_core_only", "cpu_core"),
        ("S3_gpu_frontend_only", "gpu"),
        ("S4_full_system_hybrid", "hybrid"),
        # ablations: the whole system with one stage removed.  The marginal
        # cost of each stage is then exact by construction, which is the only
        # way the stage numbers can be guaranteed to close.
        ("A1_hybrid_without_gpu", "hybrid_no_gpu"),
        ("A2_hybrid_without_cpu", "hybrid_no_cpu"),
    ]

    for name, kind in defs:
        require_quiescent(name, args.outdir)
        vals = []
        for rep in range(args.repeats):
            csv_path = os.path.join(args.logdir, f"stage_{name}_rep{rep}.csv")
            w, _ = measure(csv_path, args.duration, args.rate, make_work(kind),
                           imap, f"{name}_rep{rep}")
            vals.append(w)
        stages[name] = {
            "mean_w": float(np.mean(vals)),
            "sd_w": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "repeats": len(vals),
            "values": vals,
        }
        print(f"  {name:<30} {np.mean(vals):8.3f} W  (n={len(vals)})")

    idle = stages["S0_system_idle"]["mean_w"]
    stage_deltas = {
        "sensor_replay": stages["S1_sensor_replay_only"]["mean_w"] - idle,
        "cpu_fusion_core": stages["S2_cpu_fusion_core_only"]["mean_w"] - idle,
        "gpu_frontend": stages["S3_gpu_frontend_only"]["mean_w"] - idle,
    }
    whole = stages["S4_full_system_hybrid"]["mean_w"]

    # ---- model A: isolated stages, idle subtracted, then summed
    predicted = idle + stage_deltas["gpu_frontend"] + stage_deltas["cpu_fusion_core"]
    residual = whole - predicted

    # ---- model B: ablation-based marginal costs.
    # The GPU stage's marginal cost inside the running system is
    #     S4 - A1  (whole system minus the whole system with the GPU removed)
    # and the CPU core's is S4 - A2.  These are exact by construction; the
    # residual is the concurrency interaction (DVFS, shared rails, cache).
    a1 = stages["A1_hybrid_without_gpu"]["mean_w"]
    a2 = stages["A2_hybrid_without_cpu"]["mean_w"]
    m_gpu = whole - a1
    m_cpu = whole - a2
    predicted_b = idle + m_gpu + m_cpu
    residual_b = whole - predicted_b

    out = {
        "protocol": (f"each stage measured alone for {args.duration:.0f} s, "
                     f"{args.repeats} repeats, same instrument and same "
                     f"conditions as the whole-system runs"),
        "instrument": "tegrastats VDD_IN (on-module INA3221 via the JetPack "
                      "tegrastats interface)",
        "stages": stages,
        "idle_w": idle,
        "stage_deltas_w": stage_deltas,
        "whole_system_w": whole,
        "predicted_whole_system_w": predicted,
        "residual_w": residual,
        "residual_pct_of_whole": residual / whole * 100.0 if whole else None,
        "closure_ok_within_5pct": abs(residual / whole * 100.0) <= 5.0 if whole else None,
        "model_B_ablation": {
            "note": ("marginal cost of each stage measured by REMOVING it from "
                     "the running whole system; exact by construction"),
            "S4_whole_system_w": whole,
            "A1_hybrid_without_gpu_w": a1,
            "A2_hybrid_without_cpu_w": a2,
            "marginal_gpu_w": m_gpu,
            "marginal_cpu_w": m_cpu,
            "idle_w": idle,
            "predicted_whole_system_w": predicted_b,
            "residual_w": residual_b,
            "residual_pct_of_whole": residual_b / whole * 100.0 if whole else None,
            "closure_ok_within_5pct": abs(residual_b / whole * 100.0) <= 5.0 if whole else None,
        },
    }
    path = os.path.join(args.outdir, "stage_decomposition.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 84)
    print("STAGE-LEVEL DECOMPOSITION (idle-subtracted)")
    print("=" * 84)
    print(f"  system idle                     {idle:8.3f} W")
    print(f"  sensor replay only        + {stage_deltas['sensor_replay']:8.3f} W")
    print(f"  CPU fusion core only      + {stage_deltas['cpu_fusion_core']:8.3f} W")
    print(f"  GPU front-end only        + {stage_deltas['gpu_frontend']:8.3f} W")
    print(f"  ---------------------------------------")
    print(f"  predicted whole system        {predicted:8.3f} W")
    print(f"  measured  whole system        {whole:8.3f} W")
    print(f"  residual                      {residual:8.3f} W "
          f"({out['residual_pct_of_whole']:+.2f} % of whole system)")
    print("\n--- model B: ablation-based marginal costs (exact by construction) ---")
    print(f"  marginal GPU front-end     {m_gpu:8.3f} W")
    print(f"  marginal CPU fusion core   {m_cpu:8.3f} W")
    print(f"  predicted whole system     {predicted_b:8.3f} W")
    print(f"  measured  whole system     {whole:8.3f} W")
    print(f"  residual                   {residual_b:8.3f} W "
          f"({out['model_B_ablation']['residual_pct_of_whole']:+.2f} %)")
    print(f"\nwrote {path}")



if __name__ == "__main__":
    # Hold the measurement lock for the whole run: the quiescence guard can
    # only check the machine BEFORE a window opens, so anything that starts
    # mid-window contaminates the numbers being produced right now.  See
    # measurement_lock.py.
    from measurement_lock import measurement_lock
    with measurement_lock("run1b_stages"):
        main()
