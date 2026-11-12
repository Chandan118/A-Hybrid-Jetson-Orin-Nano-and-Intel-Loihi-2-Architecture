#!/usr/bin/env python3
"""Aggregate the quick fair-baseline pass into a reproducible summary.

The fair-baseline pass (results/run1_fairbaseline/) is the SECOND,
INDEPENDENT confirmation of the Table 1 ordering: 3 repeats x 1500 frames
per configuration, the same recorded bag and the same int8/sparse weights
as the full 5 x 30,000-frame protocol, but a shorter window.

This script re-derives results/run1_fairbaseline/fair_baseline_summary.json
from results/run1_fairbaseline/all_runs.json, so that the file is
reproducible from the raw per-repeat records instead of being an orphaned
artefact that nothing can regenerate.

Standard-deviation convention
-----------------------------
SAMPLE sd (ddof = 1), matching analyse_run1.py, the manuscript table
builder, the Table 1 aggregate and run1c3_ros2.py.  A single convention is
used everywhere so that the same repeat set cannot quote two different
+/- values in two documents.

Outputs results/run1_fairbaseline/fair_baseline_summary.json.
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "results", "run1_fairbaseline")
ALL_RUNS = os.path.join(BASE, "all_runs.json")
OUT = os.path.join(BASE, "fair_baseline_summary.json")

CONFIGS = ("C1", "C3", "C5")


def _sd(vals):
    """SAMPLE standard deviation (ddof = 1); 0.0 for a single sample."""
    if not vals:
        return None
    return st.stdev(vals) if len(vals) > 1 else 0.0


def _mean(vals):
    return st.mean(vals) if vals else None


def main():
    if not os.path.exists(ALL_RUNS):
        raise SystemExit(f"missing {ALL_RUNS}")

    with open(ALL_RUNS) as f:
        all_runs = json.load(f)

    out = {
        "protocol": {
            "repeats_per_config": 3,
            "frames_per_repeat": 1500,
            "rate_hz": 50,
            "source": "results/run1_fairbaseline/all_runs.json",
            "note": ("Quick, INDEPENDENT confirmation pass of the Table 1 "
                     "ordering. It is NOT the full 5 x 30,000-frame "
                     "protocol; the paper quotes the full-protocol figures "
                     "from results/run1/table1_aggregate.json."),
            "sd_convention": "sample sd (ddof = 1)",
        },
        "configs": {},
    }

    for cfg in CONFIGS:
        reps = all_runs.get(cfg)
        if not reps:
            continue

        pw = [r.get("mean_power_w") for r in reps if r.get("mean_power_w")]
        fps = [r.get("fps") for r in reps if r.get("fps")]
        jpf = [r.get("j_per_frame_integrated") for r in reps
               if r.get("j_per_frame_integrated")]
        ina = [r.get("ina3221_vdd_in_mean_w") for r in reps
               if r.get("ina3221_vdd_in_mean_w")]
        drop = [r.get("dropped") for r in reps if r.get("dropped") is not None]
        lat = [(r.get("latency_ms") or {}).get("p50") for r in reps]
        lat = [x for x in lat if x is not None]
        identity = [r.get("j_per_frame_agreement_pct") for r in reps
                    if r.get("j_per_frame_agreement_pct") is not None]
        # the contamination verdict is nested, under the per-repeat
        # "contamination" record written by the harness
        cont = [(r.get("contamination") or {}).get("clean") for r in reps]
        cont = [x for x in cont if x is not None]

        out["configs"][cfg] = {
            "n_repeats": len(reps),
            "power_w_mean": _mean(pw),
            "power_w_sd": _sd(pw),
            "fps_mean": _mean(fps),
            "fps_sd": _sd(fps),
            "j_per_frame_mean": _mean(jpf),
            "j_per_frame_sd": _sd(jpf),
            "dropped_mean": _mean(drop),
            "latency_p50_ms_mean": _mean(lat),
            "ina3221_vdd_in_mean_w": _mean(ina),
            "energy_identity_max_pct": max(identity) if identity else None,
            "frames": max((r.get("frames") or 0) for r in reps) if reps else None,
            "sustains_50hz": bool(fps) and all(f >= 49.9 for f in fps),
            "contamination_clean_all": bool(cont) and all(cont),
        }

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 96)
    print("FAIR-BASELINE CONFIRMATION PASS -- 3 repeats x 1500 frames "
          "(independent of the full protocol)")
    print("=" * 96)
    print(f"{'Config':<8}{'Power (W)':>22}{'INA3221 (W)':>16}{'FPS':>20}"
          f"{'J/frame':>20}")
    for cfg in CONFIGS:
        v = out["configs"].get(cfg)
        if not v:
            continue
        print(f"{cfg:<8}{v['power_w_mean']:.4f} +/- {v['power_w_sd']:.4f}      "
              f"{v['ina3221_vdd_in_mean_w']:.4f}   "
              f"{v['fps_mean']:.3f} +/- {v['fps_sd']:.3f}   "
              f"{v['j_per_frame_mean']:.6f} +/- {v['j_per_frame_sd']:.6f}")
    print(f"\n  energy identity (power/FPS vs integrated J/frame) max deviation: "
          f"{max((out['configs'][c]['energy_identity_max_pct'] for c in out['configs']), default=0):.3e} %")
    print(f"  all repeats contamination-clean: "
          f"{all(out['configs'][c]['contamination_clean_all'] for c in out['configs'])}")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
