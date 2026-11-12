#!/usr/bin/env python3
"""
Regenerate the paper's supplementary figures S1, S2 and S3 from the
MEASURED results in `exp_rev1/results/`.

Why this exists
---------------
The published figures S1, S2 and S3 were drawn from hard-coded and
`np.random`-generated values:

  * S1 ("Neuromorphic Acceleration") hard-codes the synthetic pair
    (49.73 FPS / 60.61 and 84.34 FPS / 79.32 inverse-J/frame) and states
    "+69.6 % throughput improvement and -23.6 % energy reduction";
  * S2 ("Memory Utilization") generates three entirely synthetic memory
    traces with `np.random.normal`;
  * S3 ("Thermal Profile") hard-codes (45.2, 52.1, 42.8, 48.3) degrees.

Those are the figures a reader would take the paper's performance claim
from, and none of them is a measurement.  This script redraws all three
from the measured data, and keeps the synthetic originals on disk as
`S1_synthetic.png` etc. so the change is auditable.

The measurements that feed the figures are:

  * S1 -- `exp_rev1/results/run1/summary.json` (Run 1, 5 repeats x 20 min
    per configuration, power read from the on-module INA3221);
  * S2 -- `exp_rev1/results/run2/J*_timeseries.csv` (the recorded replays,
    per-process RSS sampled during them);
  * S3 -- `exp_rev1/results/run3/summary.json` (the endurance runs, 900 s x
    3 repeats per configuration, with the maximum junction temperature).

Every figure carries a "MEASURED" label so it cannot be confused with the
synthetic originals.
"""

import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Matplotlib generates a random id for every SVG element (clip paths, marker
# definitions), so the SVG bytes change on every run even when the plotted
# content is identical.  A fixed hash salt makes the output reproducible, so
# `git diff` on the figures only ever shows a real change.
matplotlib.rcParams["svg.hashsalt"] = "jetson-orin-nano-figures"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIG = os.path.join(ROOT, "figure")

# Deterministic metadata for the SVG/PDF output.  Matplotlib otherwise
# stamps the current time into the file, so the figure bytes change on
# every run even when the plotted content is byte-identical -- which
# makes `git diff` noisy and hides real changes.
_SVG_META = {"Date": None}
# the PDF backend uses a different key for the same thing
_PDF_META = {"CreationDate": None}
R = os.path.join(HERE, "results")

C_JETSON = "#e74c3c"
C_SPLIT = "#27ae60"
C_BLUE = "#1f77b4"
C_ORANGE = "#ff7f0e"


def load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def keep_synthetic(name):
    """Preserve the synthetic original of a figure as evidence.

    The synthetic original is the figure as it was published, and it is the
    evidence for the provenance finding, so it must survive every later
    regeneration of the measured figure.  Copying the *current* file over it
    on each run would destroy it the second time the script ran: the
    measured figure would be copied over the synthetic one, and the two
    would become identical.

    So: only write the synthetic copy if it does not already exist.  If it
    does, the existing file is the original and is left alone.
    """
    for ext in (".png", ".pdf", ".svg"):
        p = os.path.join(FIG, name + ext)
        dst = os.path.join(FIG, name + "_synthetic" + ext)
        if os.path.exists(p) and not os.path.exists(dst):
            shutil.copy2(p, dst)
            print(f"  preserved the synthetic original as {os.path.basename(dst)}")
        elif os.path.exists(p) and os.path.exists(dst):
            same = (os.path.getsize(p) == os.path.getsize(dst))
            if same:
                # The two files are the same size, which for these figures
                # means the synthetic copy has been overwritten by a
                # measured one.  That is the bug this guard exists to
                # prevent, so say so loudly rather than silently accepting it.
                print(f"  WARNING: {os.path.basename(dst)} is the same size "
                      f"as {os.path.basename(p)} -- the synthetic original "
                      f"may have been overwritten.  Recover it with "
                      f"`git show <commit-before-regeneration>:figure/"
                      f"{name}{ext}`.")


def main():
    run1 = load(os.path.join(R, "run1", "summary.json"))
    run2 = load(os.path.join(R, "run2", "summary.json"), {}) or {}
    run3 = load(os.path.join(R, "run3", "summary.json"), {}) or {}
    if not run1:
        print("no Run 1 summary -- cannot redraw the figures")
        return 1

    def g(cfg, key, sub="mean"):
        s = run1.get(cfg) or {}
        v = s.get(key)
        if isinstance(v, dict):
            return v.get(sub)
        return v

    c1_fps, c4_fps = g("C1", "fps"), g("C4", "fps")
    c1_jpf, c4_jpf = g("C1", "j_per_frame_integrated"), g("C4", "j_per_frame_integrated")
    c1_pow, c4_pow = g("C1", "power_w"), g("C4", "power_w")
    if None in (c1_fps, c4_fps, c1_jpf, c4_jpf):
        print("Run 1 summary is missing the fields the figures need")
        return 1

    keep_synthetic("S1")
    keep_synthetic("S2")
    keep_synthetic("S3")

    # =====================================================  S1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(1)
    width = 0.35
    ax1.bar(x - width / 2, [c1_fps], width, label="Jetson only (C1)",
            color=C_JETSON, edgecolor="black", linewidth=1.2)
    ax1.bar(x + width / 2, [c4_fps], width, label="GPU front end + CPU core (C4)",
            color=C_SPLIT, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("Frames Per Second (FPS)", fontsize=12, fontweight="bold")
    ax1.set_title("Throughput", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["FPS"])
    ax1.legend(loc="lower left", fontsize=10)
    ax1.set_ylim(0, max(c1_fps, c4_fps) * 1.30)
    for xi, v in ((x[0] - width / 2, c1_fps), (x[0] + width / 2, c4_fps)):
        ax1.text(xi, v + max(c1_fps, c4_fps) * 0.02, f"{v:.2f}",
                 ha="center", fontsize=11, fontweight="bold")
    ax1.axhline(50, color="gray", ls="--", lw=1.2)
    ax1.text(0.62, 50.6, "50 Hz replay rate", fontsize=9, color="gray")

    inv1, inv4 = 1.0 / c1_jpf, 1.0 / c4_jpf
    ax2.bar(x - width / 2, [inv1], width, label="Jetson only (C1)",
            color=C_JETSON, edgecolor="black", linewidth=1.2)
    ax2.bar(x + width / 2, [inv4], width, label="GPU front end + CPU core (C4)",
            color=C_SPLIT, edgecolor="black", linewidth=1.2)
    ax2.set_ylabel("Energy efficiency (1 / J per frame)", fontsize=12,
                    fontweight="bold")
    ax2.set_title("Energy per frame", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["frames per joule"])
    ax2.legend(loc="lower left", fontsize=10)
    ax2.set_ylim(0, max(inv1, inv4) * 1.30)
    for xi, v in ((x[0] - width / 2, inv1), (x[0] + width / 2, inv4)):
        ax2.text(xi, v + max(inv1, inv4) * 0.02, f"{v:.2f}",
                 ha="center", fontsize=11, fontweight="bold")

    fig.suptitle("Figure S1. Measured throughput and energy per frame\n"
                 "(Jetson Orin Nano, 5 repeats x 20 min per configuration; "
                 "power from the on-module INA3221)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(FIG, f"S1.{ext}"), dpi=200,
                    bbox_inches="tight",
                    metadata=_PDF_META if ext == "pdf" else _SVG_META)
    plt.close(fig)
    print("wrote S1")

    # =====================================================  S2
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = {
        "J1": "Jetson only",
        "J2": "Jetson only + Algorithm 1",
        "J3": "GPU front end + CPU core, without Algorithm 1",
        "J4": "GPU front end + CPU core, with Algorithm 1",
    }
    plotted = 0
    for key, label in labels.items():
        p = os.path.join(R, "run2", f"{key}_timeseries.csv")
        if not os.path.exists(p):
            continue
        data = np.genfromtxt(p, delimiter=",", names=True)
        if data.size == 0:
            continue
        cols = data.dtype.names
        # the time column is literally "t"; fall back to it if the
        # descriptive heuristic does not find one
        tcol = next((c for c in cols if "t" in c.lower() and
                     ("sec" in c.lower() or "elapsed" in c.lower())), None)
        if tcol is None and "t" in cols:
            tcol = "t"
        # the memory column to plot: the SLAM process's RSS
        ycol = next((c for c in cols if c.lower() == "slam_rss_mb"), None)
        if ycol is None:
            ycol = next((c for c in cols if "rss" in c.lower() or
                         "map_live" in c.lower() or "mem" in c.lower()), None)
        if not ycol:
            continue
        t = data[tcol] if tcol else np.arange(len(data))
        t = np.asarray(t, dtype=float)
        t = t - t[0]
        if t.max() > 1000:
            t = t / 60.0
        ax.plot(t, np.asarray(data[ycol], dtype=float), lw=1.2, label=label)
        plotted += 1
    if not plotted:
        print("no Run 2 time series yet -- S2 left as the synthetic original")
    else:
        total = None
        try:
            with open("/proc/meminfo") as f:
                for ln in f:
                    if ln.startswith("MemTotal:"):
                        total = int(ln.split()[1]) / 1024.0
        except Exception:
            pass
        if total:
            ax.axhline(total, color="red", ls="--", lw=1.5,
                       label=f"Total RAM ({total:.0f} MB)")
        ax.set_xlabel("minutes since start", fontsize=12)
        ax.set_ylabel("process RSS (MB)", fontsize=12)
        ax.set_title("Figure S2. Measured memory during the recorded replays\n"
                     "(67 replays per configuration, per-process RSS sampled "
                     "throughout)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(os.path.join(FIG, f"S2.{ext}"), dpi=200,
                        metadata=_PDF_META if ext == "pdf" else _SVG_META,
                        bbox_inches="tight")
        plt.close(fig)
        print("wrote S2")

    # =====================================================  S3
    if run3:
        names, avgs, peaks = [], [], []
        for key, v in run3.items():
            if key.startswith("_") or not isinstance(v, dict):
                continue
            if "max_tj_c" not in v:
                continue
            names.append({"C1": "Jetson only", "C4": "GPU front end + CPU core"}.get(key, key))
            avgs.append(v.get("mean_power_w", 0.0))
            peaks.append(v.get("max_tj_c", 0.0))
        if names:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                           gridspec_kw={"height_ratios": [3, 1]})
            xi = np.arange(len(names))
            ax1.bar(xi - 0.2, avgs, 0.4, label="mean power (W)",
                    color=C_BLUE, edgecolor="black")
            ax1.set_ylabel("Power (W)", fontsize=12)
            ax1.set_xticks(xi)
            ax1.set_xticklabels(names)
            ax1.legend(fontsize=10)
            ax1.grid(alpha=0.3, axis="y")
            ax1.set_title("Figure S3. Measured endurance\n"
                          "(900 s x 3 repeats per configuration, continuous "
                          "sensor replay)", fontsize=13, fontweight="bold")
            ax2.bar(xi, peaks, 0.5, color=C_ORANGE, edgecolor="black")
            ax2.set_ylabel("max junction temp (°C)", fontsize=12)
            ax2.set_xticks(xi)
            ax2.set_xticklabels(names)
            for x0, v in zip(xi, peaks):
                ax2.text(x0, v, f"{v:.1f}°C", ha="center", va="bottom",
                         fontsize=10, fontweight="bold")
            ax2.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            for ext in ("png", "pdf", "svg"):
                fig.savefig(os.path.join(FIG, f"S3.{ext}"), dpi=200,
                            metadata=_PDF_META if ext == "pdf" else _SVG_META,
                            bbox_inches="tight")
            plt.close(fig)
            print("wrote S3")

    print("\nThe previous synthetic figures were kept as S*_synthetic.*")

    # ------------------------------------------------- verify the evidence
    # The synthetic originals are the evidence for the provenance finding,
    # so they must still differ from the measured figures after this run.
    # If they are identical, the synthetic original has been lost and the
    # audit trail is broken -- say so rather than reporting success.
    print("\nverifying the synthetic originals survived:")
    ok = True
    for name in ("S1", "S2", "S3"):
        a = os.path.join(FIG, name + ".png")
        b = os.path.join(FIG, name + "_synthetic.png")
        if not (os.path.exists(a) and os.path.exists(b)):
            print(f"  {name}: measured/synthetic pair not both present")
            ok = False
            continue
        if open(a, "rb").read() == open(b, "rb").read():
            print(f"  {name}: IDENTICAL -- the synthetic original has been "
                  f"overwritten; recover it from git before publishing")
            ok = False
        else:
            print(f"  {name}: measured and synthetic differ (good)")
    if not ok:
        print("\nFAILED: the synthetic originals are not intact.")
        return 1
    return 0



if __name__ == "__main__":
    # This tool is CPU-heavy and must NOT run while a measurement is in
    # progress: it would contaminate the open measurement window.  On
    # 2026-11-05 it did exactly that and cost a whole 30 Hz pass.
    from measurement_lock import refuse_if_measurement_running
    refuse_if_measurement_running("make_paper_figures.py")
    raise SystemExit(main())
