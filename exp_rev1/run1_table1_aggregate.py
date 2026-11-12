#!/usr/bin/env python3
"""Aggregate the FULL-PROTOCOL baseline repeats into the Table 1 numbers.

The fair-baseline rerun (results/run1_fairbaseline/) is a QUICK pass: 3
repeats x 1500 frames.  The original protocol (results/run1/repeats/) is
the stronger data set and is already complete for C1/C3/C5: 5 repeats x
30,000 frames each (10 min warm-up + 10 min log), 12,001 power samples per
repeat, contamination-clean in every repeat.

This script re-derives the Table 1 numbers from the RAW tegrastats CSVs
(logs/run1/<cfg>_rep<N>.csv) rather than from the harness's own summary,
so that the power figure in the paper can be traced to the instrument
samples, and it cross-checks the two independent power sources:

  teg_vdd_in_mw_avg  -- the module's VDD_IN rail, from tegrastats
  ina_vdd_in_w       -- the same rail, from the on-module INA3221

Outputs results/run1/table1_aggregate.json.

Standard-deviation convention
-----------------------------
The spread reported across the repeats is the SAMPLE standard deviation
(ddof = 1), matching the harness's own aggregation (analyse_run1.py),
the manuscript table builder, the figures, the appeal letters and the
review response.  A single convention is used everywhere so that the
same repeat set cannot quote two different +/- values in two documents.
"""
import csv
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPEATS = os.path.join(HERE, "results", "run1", "repeats")
LOGS = os.path.join(HERE, "logs", "run1")
OUT = os.path.join(HERE, "results", "run1", "table1_aggregate.json")

CONFIGS = ("C1", "C3", "C5")
N_REPEATS = 5


def read_power_csv(path):
    """Mean/std of both power instruments over the logged window."""
    teg, ina = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row.get("teg_vdd_in_mw_avg")
            if v:
                teg.append(float(v) / 1000.0)
            v = row.get("ina_vdd_in_w")
            if v:
                ina.append(float(v))
    return teg, ina


def main():
    out = {"protocol": {
        "warmup_s": 600, "log_s": 600, "frames_per_repeat": 30000,
        "rate_hz": 50, "repeats_per_config": N_REPEATS,
        "power_sample_rate_hz": 20,
        "source_csvs": "logs/run1/<cfg>_rep<N>.csv",
        "note": ("Full protocol. The quick 3x1500 fair-baseline pass "
                 "(results/run1_fairbaseline/) is retained separately as a "
                 "second, shorter confirmation of the same ordering."),
    }, "configs": {}}

    for cfg in CONFIGS:
        reps = []
        for rep in range(N_REPEATS):
            jp = os.path.join(REPEATS, f"{cfg}_rep{rep}.json")
            cp = os.path.join(LOGS, f"{cfg}_rep{rep}.csv")
            if not (os.path.exists(jp) and os.path.exists(cp)):
                continue
            d = json.load(open(jp))
            r = d.get("run", d)
            teg, ina = read_power_csv(cp)
            cont = r.get("contamination") or {}
            reps.append({
                "rep": rep,
                "frames": r.get("frames"),
                "dropped": r.get("dropped"),
                "elapsed_s": r.get("elapsed_s"),
                "fps": r.get("fps"),
                "energy_j": r.get("energy_j"),
                "j_per_frame": r.get("j_per_frame_integrated"),
                "j_per_frame_agreement_pct": r.get("j_per_frame_agreement_pct"),
                "latency_p50_ms": (r.get("latency_ms") or {}).get("p50"),
                "latency_p95_ms": (r.get("latency_ms") or {}).get("p95"),
                # NOTE: this is the spread of the 12,001 individual
                # tegrastats samples WITHIN one repeat (instrument noise
                # over the logged window).  It is NOT the across-repeat
                # spread quoted in Table 1, and it is a population sd
                # because the window's samples are the complete set.
                "tegrastats_vdd_in_mean_w": st.mean(teg) if teg else None,
                "tegrastats_vdd_in_sd_w": st.pstdev(teg) if teg else None,
                "tegrastats_vdd_in_samples": len(teg),
                "ina3221_vdd_in_mean_w": st.mean(ina) if ina else None,
                "ina3221_vdd_in_sd_w": st.pstdev(ina) if ina else None,
                "contamination_clean": cont.get("clean"),
                "contamination_count": len(cont.get("contaminants") or []),
                "gpu_clock_hz": r.get("gpu_clock_hz"),
            })

        if not reps:
            continue

        def agg(key):
            """Across-repeat spread: SAMPLE sd (ddof = 1)."""
            vals = [x[key] for x in reps if x.get(key) is not None]
            return (st.mean(vals), st.stdev(vals)) if len(vals) > 1 \
                else ((st.mean(vals), 0.0) if vals else (None, None))

        p_mean, p_sd = agg("tegrastats_vdd_in_mean_w")
        i_mean, i_sd = agg("ina3221_vdd_in_mean_w")
        fps_mean, fps_sd = agg("fps")
        j_mean, j_sd = agg("j_per_frame")
        drop_mean, _ = agg("dropped")
        lat_mean, _ = agg("latency_p50_ms")
        lat95_mean, _ = agg("latency_p95_ms")

        out["configs"][cfg] = {
            "n_repeats": len(reps),
            "repeats": reps,
            "power_w_mean": p_mean, "power_w_sd": p_sd,
            "ina3221_power_w_mean": i_mean, "ina3221_power_w_sd": i_sd,
            "fps_mean": fps_mean, "fps_sd": fps_sd,
            "j_per_frame_mean": j_mean, "j_per_frame_sd": j_sd,
            "dropped_mean": drop_mean,
            "latency_p50_ms_mean": lat_mean,
            "latency_p95_ms_mean": lat95_mean,
            "sustains_50hz_all": all(f >= 49.9 for f in
                                      (x["fps"] for x in reps) if f),
            "contamination_clean_all": all(x["contamination_clean"] for x in reps),
            "energy_identity_max_pct": max(
                (x["j_per_frame_agreement_pct"] for x in reps
                 if x.get("j_per_frame_agreement_pct") is not None), default=None),
        }

    # derived comparison
    c = out["configs"]
    if "C1" in c and "C3" in c:
        out["comparison"] = {
            "C3_vs_C1_energy_per_frame_saving_percent":
                (1 - c["C3"]["j_per_frame_mean"] / c["C1"]["j_per_frame_mean"]) * 100,
            "C3_vs_C1_power_saving_percent":
                (1 - c["C3"]["power_w_mean"] / c["C1"]["power_w_mean"]) * 100,
            "C5_vs_C1_energy_per_frame_saving_percent":
                (1 - c["C5"]["j_per_frame_mean"] / c["C1"]["j_per_frame_mean"]) * 100,
        }

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 104)
    print("TABLE 1 -- full protocol, 5 repeats x 30,000 frames, power at the Jetson VDD_IN rail")
    print("=" * 104)
    print(f"{'Config':<8}{'Power (W)':>22}{'INA3221 (W)':>16}{'FPS':>18}"
          f"{'J/frame':>20}{'dropped':>10}{'50Hz':>6}")
    for cfg in CONFIGS:
        if cfg not in c:
            continue
        v = c[cfg]
        print(f"{cfg:<8}{v['power_w_mean']:.4f} +/- {v['power_w_sd']:.4f}      "
              f"{v['ina3221_power_w_mean']:.4f}   "
              f"{v['fps_mean']:.3f} +/- {v['fps_sd']:.3f}   "
              f"{v['j_per_frame_mean']:.6f} +/- {v['j_per_frame_sd']:.6f}   "
              f"{v['dropped_mean']:>8.1f}   {'yes' if v['sustains_50hz_all'] else 'no':>6}")
    print()
    for k, val in (out.get("comparison") or {}).items():
        print(f"  {k}: {val:.2f} %")
    print(f"\n  energy identity (power/FPS vs integrated J/frame) max deviation: "
          f"{max((c[x]['energy_identity_max_pct'] for x in c), default=0):.3e} %")
    print(f"  all repeats contamination-clean: "
          f"{all(c[x]['contamination_clean_all'] for x in c)}")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
