#!/usr/bin/env python3
"""
Analyse the Run 1 outputs:

  * mean +/- SD power per configuration (across the 5 repeats)
  * FPS, latency p50/p95/p99, J/frame
  * the two independent J/frame computations (integrated power / frames vs
    mean power / FPS) and their agreement
  * the pose RMSE cross-check between C1..C4
"""

import os
import sys
import json
import csv
import argparse
import statistics
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_runs(rundir):
    out = {}
    for fn in sorted(os.listdir(rundir)):
        if fn.endswith("_run.json"):
            with open(os.path.join(rundir, fn)) as f:
                d = json.load(f)
            cfg = d["runs"][0]["config"]
            out[cfg] = d
    return out


def agg(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals)}


def pose_rmse(a, b):
    """RMSE between two pose sequences, aligned in time."""
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    a = np.asarray(a[:n], dtype=float)
    b = np.asarray(b[:n], dtype=float)
    d = np.sqrt(((a[:, 0] - b[:, 0]) ** 2 + (a[:, 1] - b[:, 1]) ** 2))
    return float(np.sqrt(np.mean(d ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", default=os.path.join(HERE, "results", "run1"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = load_runs(args.rundir)
    summary = {}

    print("=" * 100)
    print("RUN 1 -- POWER AND ENERGY")
    print("=" * 100)

    rows = []
    for cfg, d in runs.items():
        rs = d["runs"]
        s = {
            "config": cfg,
            "power_w": agg([r["mean_power_w"] for r in rs]),
            "fps": agg([r["fps"] for r in rs]),
            "j_per_frame_integrated": agg([r["j_per_frame_integrated"] for r in rs]),
            "j_per_frame_mean_over_fps": agg([r["j_per_frame_mean_over_fps"] for r in rs]),
            "j_per_frame_agreement_pct_max": max(r["j_per_frame_agreement_pct"] for r in rs),
            "latency_p50_ms": agg([r["latency_ms"]["p50"] for r in rs]),
            "latency_p95_ms": agg([r["latency_ms"]["p95"] for r in rs]),
            "latency_p99_ms": agg([r["latency_ms"]["p99"] for r in rs]),
            "latency_max_ms": agg([r["latency_ms"]["max"] for r in rs]),
            "vdd_soc_w": agg([r["vdd_soc_mean_w"] for r in rs]),
            "vdd_cvgcv_w": agg([r["vdd_cvgcv_mean_w"] for r in rs]),
            "ina3221_vdd_in_w": agg([r["ina3221_vdd_in_mean_w"] for r in rs]),
            "frames_per_run": rs[0]["frames"],
            "dropped_per_run": [r["dropped"] for r in rs],
        }
        summary[cfg] = s
        rows.append(s)

    print(f"\n{'config':<34}{'power W (mean±SD)':>24}{'FPS':>16}{'J/frame':>18}{'agree %':>10}")
    print("-" * 104)
    for s in rows:
        p, f, j = s["power_w"], s["fps"], s["j_per_frame_integrated"]
        print(f"{s['config']:<34}"
              f"{p['mean']:>12.3f} ± {p['sd']:<8.3f}"
              f"{f['mean']:>9.2f} ± {f['sd']:<5.2f}"
              f"{j['mean']:>10.5f} ± {j['sd']:<6.5f}"
              f"{s['j_per_frame_agreement_pct_max']:>10.4f}")

    print(f"\n{'config':<34}{'latency p50 ms':>18}{'p95 ms':>13}{'p99 ms':>13}{'max ms':>13}")
    print("-" * 100)
    for s in rows:
        print(f"{s['config']:<34}"
              f"{s['latency_p50_ms']['mean']:>10.3f} ± {s['latency_p50_ms']['sd']:<6.3f}"
              f"{s['latency_p95_ms']['mean']:>11.3f} {s['latency_p99_ms']['mean']:>11.3f} "
              f"{s['latency_max_ms']['mean']:>11.3f}")

    print("\n" + "=" * 100)
    print("POSE CROSS-CHECK (C1..C4 must produce the same output)")
    print("=" * 100)
    ref_cfg = "C1_original_gpu_pipeline" if "C1_original_gpu_pipeline" in runs else list(runs)[0]
    ref_pose = runs[ref_cfg]["runs"][0]["poses"]
    pose_table = {}
    for cfg, d in runs.items():
        p = d["runs"][0]["poses"]
        rm = pose_rmse(ref_pose, p)
        fin = p[-1] if p else (0, 0, 0)
        pose_table[cfg] = {"pose_rmse_vs_ref_m": rm,
                           "final_pose": list(fin),
                           "ref_config": ref_cfg,
                           "ref_final_pose": list(ref_pose[-1]) if ref_pose else None}
        print(f"  {cfg:<36} pose RMSE vs {ref_cfg}: {rm:.6f} m   "
              f"final=({fin[0]:.5f}, {fin[1]:.5f}, {fin[2]:.5f})")
    summary["_pose_crosscheck"] = pose_table
    summary["_pose_reference_config"] = ref_cfg

    out = args.out or os.path.join(args.rundir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
