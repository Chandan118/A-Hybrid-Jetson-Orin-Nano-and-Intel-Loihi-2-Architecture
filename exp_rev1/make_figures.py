#!/usr/bin/env python3
"""
Generate the figures of the revision response from the MEASURED results.

  fig_power.png     mean power per configuration with the repeat SD
  fig_jpf.png       joules per frame per configuration (both computations)
  fig_latency.png   latency p50 / p95 / p99 per configuration
  fig_map_growth.png  map size over time, compressed vs uncompressed
  fig_memory.png    peak per-component RSS per configuration
"""

import os
import json
import statistics

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
FIG = os.path.join(os.path.dirname(HERE), "figure")
os.makedirs(FIG, exist_ok=True)

LABEL = {
    "C1": "C1 GPU\n(PyTorch)",
    "C2": "C2 TensorRT\n(FP32)",
    "C3": "C3 C++/ARM\n(int8 sparse)",
    "C4": "C4 hybrid\n(GPU+CPU core)",
    # C5 is MEASURED on the Torch CPU path: the Lava neuromorphic runtime
    # is not installed on this machine and the use_lava flag is never set.
    # See Lookup A in the reviewer response.
    "C5": "C5 CPU\n(Torch CPU)",
}


def load(p):
    q = os.path.join(R, p)
    if not os.path.exists(q):
        return None
    with open(q) as f:
        return json.load(f)


def fig_power(runs):
    keys = sorted(runs.keys(), key=lambda k: int(k.split("_")[0][1:]))
    means, sds, labels = [], [], []
    for k in keys:
        rs = runs[k]
        p = [r["mean_power_w"] for r in rs if r.get("mean_power_w")]
        means.append(statistics.mean(p))
        sds.append(statistics.stdev(p) if len(p) > 1 else 0.0)
        labels.append(LABEL.get(k.split("_")[0], k))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, means, yerr=sds, capsize=5,
                  color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"])
    for b, m, s in zip(bars, means, sds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.15,
                f"{m:.2f}±{s:.2f} W", ha="center", fontsize=9)
    ax.set_ylabel("Mean power at the Jetson VDD_IN input (W)")
    ax.set_title("Run 1 — measured power per configuration\n"
                 "(5 repeats, 10 min warm-up + 10 min log each)", fontsize=11)
    ax.set_ylim(0, max(means) + max(sds) * 2 + 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "R1_power.png"), dpi=160)
    plt.close(fig)


def fig_jpf(runs):
    keys = sorted(runs.keys(), key=lambda k: int(k.split("_")[0][1:]))
    a = [statistics.mean([r["j_per_frame_integrated"] for r in runs[k]]) for k in keys]
    b = [statistics.mean([r["j_per_frame_mean_over_fps"] for r in runs[k]]) for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - 0.2, a, 0.4, label="∫P dt / frames", color="#4C72B0")
    ax.bar(x + 0.2, b, 0.4, label="mean P / FPS", color="#55A868")
    for xi, (va, vb) in enumerate(zip(a, b)):
        ax.text(xi - 0.2, va, f"{va:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + 0.2, vb, f"{vb:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(k.split("_")[0], k) for k in keys])
    ax.set_ylabel("Joules per frame (J)")
    ax.set_title("Run 1 — energy per frame by two independent computations")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "R1_jpf.png"), dpi=160)
    plt.close(fig)


def fig_latency(runs):
    keys = sorted(runs.keys(), key=lambda k: int(k.split("_")[0][1:]))
    p50 = [statistics.mean([r["latency_ms"]["p50"] for r in runs[k]]) for k in keys]
    p95 = [statistics.mean([r["latency_ms"]["p95"] for r in runs[k]]) for k in keys]
    p99 = [statistics.mean([r["latency_ms"]["p99"] for r in runs[k]]) for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, p50, "o-", label="p50")
    ax.plot(x, p95, "s-", label="p95")
    ax.plot(x, p99, "^-", label="p99")
    for xi, v in zip(x, p50):
        ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(k.split("_")[0], k) for k in keys])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Run 1 — per-frame latency distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "R1_latency.png"), dpi=160)
    plt.close(fig)


def fig_map_growth():
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, key in zip(axes, ("J1", "J2", "J3", "J4")):
        p = os.path.join(R, "run2", f"{key}_timeseries.csv")
        if not os.path.exists(p):
            ax.set_visible(False)
            continue
        import csv as _csv
        with open(p) as f:
            rows = list(_csv.DictReader(f))
        t = [(float(r["t"]) - float(rows[0]["t"])) / 60.0 for r in rows]
        live = [float(r["map_bytes_live"]) / 1e6 for r in rows]
        unc = [float(r["map_bytes_uncompressed"]) / 1e6 for r in rows]
        ax.plot(t, unc, color="#C44E52", lw=1.4, label="uncompressed")
        ax.plot(t, live, color="#55A868", lw=1.4, label="compressed (live)")
        ax.set_title(key)
        ax.set_xlabel("minutes since start")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("map size (MB)")
    fig.suptitle("Run 2 — map size over time, compressed vs uncompressed "
                 "(67 recorded replays per configuration)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "R2_map_growth.png"), dpi=160)
    plt.close(fig)


def fig_memory(run2):
    keys = [k for k in ("J1", "J2", "J3", "J4") if k in run2]
    comps = ["slam", "lidar", "map", "ros2"]
    names = ["SLAM", "LiDAR", "map module", "ROS 2"]
    x = np.arange(len(keys))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, (c, n) in enumerate(zip(comps, names)):
        vals = [run2[k]["peak_by_component_rss_mb"].get(c, 0.0) for k in keys]
        ax.bar(x + (i - 1.5) * width, vals, width, label=n)
    ax.set_xticks(x)
    ax.set_xticklabels([run2[k]["label"].replace(" + ", " +\n").replace(" without ", "\nwithout ")
                        for k in keys], fontsize=8)
    ax.set_ylabel("peak RSS (MB)")
    ax.set_title("Run 2 — peak per-component resident memory")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "R2_memory.png"), dpi=160)
    plt.close(fig)


def main():
    runs = load("run1/all_runs.json")
    if runs:
        fig_power(runs)
        fig_jpf(runs)
        fig_latency(runs)
        print("wrote R1_power.png, R1_jpf.png, R1_latency.png")
    run2 = load("run2/summary.json")
    if run2:
        fig_map_growth()
        fig_memory(run2)
        print("wrote R2_map_growth.png, R2_memory.png")


if __name__ == "__main__":
    main()
