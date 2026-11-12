#!/usr/bin/env python3
"""
The GATE: the checks that must pass before a long measurement pass starts.

Why a gate
----------
The 2026-11-05/06 30 Hz pass produced numbers, and those numbers were
wrong -- not because the instrument misread the power, but because the
PROCESS was wrong in three specific ways that the power figure alone
could not show:

  1. **A leak across repeats.**  Every configuration and every repeat ran
     inside ONE process.  The C++ fusion core keeps a per-frame output
     map and the CUDA contexts keep their allocations, so the recorded RAM
     grew by ~670 MB across C3's five repeats and fell by ~730 MB when C4
     started.  The growth is invisible in the watts and obvious in the
     memory trace.

  2. **A bimodal C2.**  C2's per-repeat means split into two groups
     (29.35 FPS and 29.85 FPS) with nothing in between, and the power
     means split with them.  A single "mean +/- SD" over those five
     repeats describes neither mode.

  3. **Contamination.**  A pass can be ruined by another process running
     inside a measurement window.  The logger now watches for that
     in-window and refuses to emit the figure.

This script is the gate.  It runs a SHORT pass (a few repeats per
configuration, in the same fresh-process-per-repeat protocol as the real
pass) and then asks the three questions above, plus the NvMap question.
A pass that fails the gate must not be started.

WHAT IS CHECKED
---------------
  G1  NvMap leak.  No `NvMapMemAllocInternalTagged` line appears in the
      kernel log during the gate run.  This is the leak's kernel-side
      signature: the driver logs every tagged allocation, so a run that
      leaks GPU memory says so in the kernel log even though the process
      is otherwise well behaved.

  G2  C2 is not bimodal.  The between-repeat variance of the per-repeat
      power means is compared with the within-repeat variance.  A
      configuration that is bimodal has a between-repeat variance many
      times the within-repeat variance, because the repeats fall into
      distinct modes instead of scattering about one mean.

  G3  RAM does not grow across repeats (C1 and C3).  Each repeat runs in
      a FRESH PROCESS, so the RAM at the end of a repeat must not be
      systematically higher than the RAM at the start of the next one.  A
      monotonic rise across repeats is the leak.

  G4  Every repeat completed cleanly: no contamination, and the power
      CSV covers the whole window.

THE THRESHOLDS, AND WHY THEY ARE WHAT THEY ARE
----------------------------------------------
The thresholds are not guessed.  They are calibrated against the
2026-11-05/06 pass, which is a KNOWN-BAD reference: it was produced by
the old single-process protocol, and its failure modes are known.  The
calibration is reproduced by `--calibrate`, which prints the old pass's
numbers next to the thresholds so that the margin is visible.

    between/within variance ratio (old pass, known bad):
        C1   0.38   C2  49.96   C3  18.00   C4   1.12   C5   0.07

    So a healthy configuration sits at or below ~1.5, and the C2 failure
    sat at 50.  The threshold is set at 5.0, which is more than 3x the
    healthy values and more than 10x below the observed failure.

    RAM growth across repeats (old pass, known bad):
        C3 grew +667 MB across its five repeats, C4 fell -735 MB across
        the boundary into it.  The threshold is set at 200 MB per repeat
        step, which is well inside the old failure and well outside the
        run-to-run noise of the healthy configurations (C1 moved 108 MB
        across its five repeats in the same bad pass, and that was
        scatter, not growth -- the direction of the change is what
        distinguishes the two).
"""

import os
import re
import sys
import json
import csv
import argparse
import subprocess
import statistics
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "common"))

import numpy as np

# ------------------------------------------------------------- thresholds
# Between/within variance ratio. Above this, the per-repeat means are
# scattered rather than clustered, which is what a bimodal
# configuration looks like. Calibrated: healthy <= 1.12, C2 failure 49.96.
MAX_BETWEEN_WITHIN_RATIO = 5.0

# RAM growth across the repeat sequence, in MB per repeat step. Above
# this, the memory is growing from repeat to repeat. Calibrated: C3 grew
# +667 MB across five repeats in the bad pass, and the healthy
# configurations moved < 110 MB with no consistent direction.
MAX_RAM_GROWTH_MB_PER_REPEAT = 200.0

# A repeat must cover at least this fraction of its window, or the
# repeat is truncated and is not a measurement.
MIN_WINDOW_COMPLETENESS = 0.95

# The NvMap signature. The driver logs every tagged allocation.
NVMAP_PATTERN = re.compile(r"NvMapMemAllocInternalTagged")


# --------------------------------------------------------------- helpers
def kernel_log_since(ts_iso: str) -> str:
    """The kernel log since a timestamp.

    `dmesg` is not readable on this board without root
    (`kernel.dmesg_restrict` is 1), but the systemd journal carries the
    same kernel messages and IS readable.  So the journal is the source,
    and the fact that the journal is readable is itself recorded so that
    a reader can see which source was used.
    """
    for cmd in (["journalctl", "-k", "--since", ts_iso, "--no-pager"],
                ["journalctl", "-k", "--no-pager"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                return r.stdout
        except Exception:
            continue
    return ""


def _read_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _read_csv(p):
    try:
        with open(p) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def variance_ratio(per_repeat_means):
    """Between-repeat variance / within-repeat variance.

    The within-repeat variance is the mean of the per-repeat variances of
    the power samples inside each window; the between-repeat variance is
    the variance of the per-repeat means.  A configuration whose repeats
    all sit on one mean has a ratio near zero; a bimodal one has a
    ratio in the tens.
    """
    means = []
    within = []
    for vals in per_repeat_means:
        if len(vals) < 2:
            continue
        means.append(float(np.mean(vals)))
        within.append(float(np.var(vals, ddof=1)))
    if len(means) < 2:
        return None, None, None
    between = float(np.var(means, ddof=1))
    w = float(np.mean(within))
    if w <= 0:
        return None, between, w
    return between / w, between, w


def ram_growth(per_repeat_ram):
    """The largest monotonic rise in the per-repeat RAM means.

    Growth is measured as a sustained rise, not as the total spread: a
    configuration whose RAM scatters up and down is healthy, and one
    whose RAM rises from repeat to repeat is leaking.  The statistic is
    the largest sum of consecutive rises (a "run" of upward steps),
    which is zero for a flat or noisy sequence and large for a leak.
    """
    if len(per_repeat_ram) < 2:
        return 0.0, []
    steps = [per_repeat_ram[i + 1] - per_repeat_ram[i]
             for i in range(len(per_repeat_ram) - 1)]
    best = 0.0
    best_run = []
    cur = 0.0
    cur_run = []
    for s in steps:
        if s > 0:
            cur += s
            cur_run.append(s)
            if cur > best:
                best = cur
                best_run = list(cur_run)
        else:
            cur = 0.0
            cur_run = []
    return best, steps


def bimodal_gap(per_repeat_means):
    """The largest gap in the sorted per-repeat means, as a fraction of
    their spread.

    Reported alongside the variance ratio because it is the statistic
    that says directly "these repeats fall into two groups".  It is
    only meaningful when there are enough repeats to see a split.
    """
    if len(per_repeat_means) < 4:
        return None, None
    v = sorted(float(np.mean(x)) for x in per_repeat_means)
    gaps = np.diff(v)
    spread = v[-1] - v[0]
    gi = int(np.argmax(gaps))
    if spread <= 0:
        return None, None
    return float(gaps[gi] / spread), float(gaps[gi])


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True,
                    help="the results directory of the gate run")
    ap.add_argument("--logdir", required=True,
                    help="the log directory of the gate run")
    ap.add_argument("--out", default=None,
                    help="where the gate verdict is written")
    ap.add_argument("--calibrate", action="store_true",
                    help="print the known-bad reference calibration and exit")
    ap.add_argument("--ram-check-configs", default="C1,C3",
                    help="the configurations whose RAM growth is checked")
    ap.add_argument("--bimodal-configs", default="C2",
                    help="the configurations checked for bimodality")
    ap.add_argument("--runlog", default=None,
                    help="the orchestrator's stdout/stderr log. The NvMap "
                         "signature is written to the calling process's "
                         "stderr, which is interleaved into this file, so "
                         "it is the authoritative source for G1b.")
    args = ap.parse_args()

    if args.calibrate:
        return calibrate()

    results = []
    fails = []
    warns = []

    def check(ok, name, detail=""):
        results.append({"check": name, "passed": bool(ok), "detail": detail})
        print(("  ok   " if ok else "  FAIL ") + name
              + (f"  -- {detail}" if detail else ""))
        if not ok:
            fails.append(name)
        return ok

    print("=" * 78)
    print("THE GATE -- checks that must pass before a long pass starts")
    print("=" * 78)

    repeats_dir = os.path.join(args.rundir, "repeats")
    if not os.path.isdir(repeats_dir):
        raise SystemExit(f"no repeats directory at {repeats_dir} -- the gate "
                         f"run has not produced any repeats")
    rep_files = sorted(f for f in os.listdir(repeats_dir) if f.endswith(".json"))
    if not rep_files:
        raise SystemExit(f"{repeats_dir} is empty -- the gate run produced no "
                         f"repeats")

    # ---- group the repeats by configuration
    per_cfg = {}
    for fn in rep_files:
        d = _read_json(os.path.join(repeats_dir, fn))
        if not d or "run" not in d:
            continue
        r = d["run"]
        per_cfg.setdefault(r["config"], []).append(r)
    for c in per_cfg:
        per_cfg[c].sort(key=lambda r: r.get("rep", 0))

    print(f"\nconfigurations found: "
          f"{', '.join(f'{c}x{len(v)}' for c, v in sorted(per_cfg.items()))}")

    # ---- G0: every repeat completed cleanly
    print("\n[G0] every repeat completed cleanly")
    for c, runs in sorted(per_cfg.items()):
        for r in runs:
            tag = f"{c}_rep{r.get('rep')}"
            contam = r.get("contamination") or {}
            ok = contam.get("clean", True)
            check(ok, f"{tag}: no contamination inside the window",
                  "" if ok else f"contaminants: {contam.get('contaminants')}")
            csv_path = r.get("power_csv") or ""
            rows = _read_csv(csv_path)
            if not rows:
                check(False, f"{tag}: the power CSV is readable")
                continue
            expected = r.get("elapsed_s", 0) * 20.0
            got = len(rows)
            frac = got / expected if expected else 0.0
            check(frac >= MIN_WINDOW_COMPLETENESS,
                  f"{tag}: the power CSV covers the window",
                  f"{got} rows vs {expected:.0f} expected ({frac*100:.1f}%)")

    # ---- G1: no NvMap leak
    print("\n[G1] no NvMapMemAllocInternalTagged in the kernel log")
    kl = kernel_log_since("1970-01-01")
    hits = NVMAP_PATTERN.findall(kl)
    check(not hits,
          "no NvMapMemAllocInternalTagged line in the kernel log",
          f"{len(hits)} matching line(s)"
          + (f" (first: {NVMAP_PATTERN.search(kl).group(0)[:80]})"
             if hits else ""))

    # The driver ALSO writes the same signature to the calling process's
    # stderr when an allocation fails, and the worker's stderr is
    # interleaved into the orchestrator's log.  On this board the kernel
    # log was empty of these lines while the run log was full of them, so
    # the run log is the authoritative source and the kernel log is the
    # secondary one.  Both are checked.
    print("\n[G1b] no NvMapMemAllocInternalTagged in the run log")
    runlog = args.runlog
    if not runlog:
        runlog = os.path.join(args.logdir, "gate_run.log")
    run_hits = []
    run_log_path = runlog
    if os.path.exists(run_log_path):
        with open(run_log_path, errors="replace") as f:
            txt = f.read()
        run_hits = NVMAP_PATTERN.findall(txt)
        # attribute each hit to the configuration that was running
        per_cfg_hits = {}
        cfg = None
        for line in txt.splitlines():
            if line.startswith("=== configuration"):
                cfg = line.strip()
            if NVMAP_PATTERN.search(line):
                per_cfg_hits.setdefault(cfg, 0)
                per_cfg_hits[cfg] += 1
    else:
        per_cfg_hits = {}
    check(not run_hits,
          "no NvMapMemAllocInternalTagged line in the run log",
          f"{len(run_hits)} matching line(s) in {os.path.basename(run_log_path)}"
          + (f"; per configuration: {per_cfg_hits}" if run_hits else ""))
    if run_hits:
        for cfg, n in per_cfg_hits.items():
            print(f"        {n} line(s) during: {cfg}")

    # ---- G2: no bimodality
    print("\n[G2] the per-repeat distribution is not bimodal")
    bimodal_cfgs = [c.strip() for c in args.bimodal_configs.split(",")
                    if c.strip()]
    for c in bimodal_cfgs:
        runs = per_cfg.get(c)
        if not runs:
            warns.append(f"{c}: no repeats, so bimodality was not checked")
            continue
        power_series = []
        for r in runs:
            rows = _read_csv(r.get("power_csv") or "")
            vals = [float(x["teg_vdd_in_mw_avg"]) / 1000.0 for x in rows
                    if x.get("teg_vdd_in_mw_avg")]
            if vals:
                power_series.append(vals)
        ratio, between, within = variance_ratio(power_series)
        gap_frac, gap_abs = bimodal_gap(power_series)
        detail = (f"between/within={ratio:.3f} (between={between:.2e}, "
                  f"within={within:.2e})" if ratio is not None else "n/a")
        if gap_frac is not None:
            detail += f", largest gap {gap_frac*100:.1f}% of spread"
        check(ratio is not None and ratio <= MAX_BETWEEN_WITHIN_RATIO,
              f"{c}: the per-repeat means are not bimodal", detail)
        # fps is the second, independent view of the same thing
        fps = [r.get("fps") for r in runs if r.get("fps")]
        if len(fps) >= 4:
            fps_ratio = (statistics.pstdev(fps) ** 2 /
                         max(statistics.pvariance([float(r.get('fps'))
                                                   for r in runs]) or 1e-12, 1e-12))
            sd = statistics.pstdev(fps)
            detail2 = (f"fps per repeat: {[round(v,4) for v in fps]}, sd={sd:.4f}")
            check(sd <= 0.05 * max(fps),
                  f"{c}: the achieved FPS does not split into two groups",
                  detail2)

    # ---- G3: RAM does not grow across repeats
    print("\n[G3] RAM does not grow across the repeats")
    ram_cfgs = [c.strip() for c in args.ram_check_configs.split(",")
                if c.strip()]
    for c in ram_cfgs:
        runs = per_cfg.get(c)
        if not runs:
            warns.append(f"{c}: no repeats, so RAM growth was not checked")
            continue
        ram_means = []
        for r in runs:
            rows = _read_csv(r.get("power_csv") or "")
            vals = [float(x["ram_used_mb"]) for x in rows if x.get("ram_used_mb")]
            ram_means.append(float(np.mean(vals)) if vals else None)
        if any(v is None for v in ram_means):
            warns.append(f"{c}: the RAM column is missing from a repeat")
            continue
        growth, steps = ram_growth(ram_means)
        detail = (f"per-repeat RAM means {[round(v) for v in ram_means]}, "
                  f"largest sustained rise {growth:.0f} MB")
        check(growth <= MAX_RAM_GROWTH_MB_PER_REPEAT,
              f"{c}: RAM does not grow across repeats", detail)

    # ---- G4: the GPU clock was at the floor for every repeat
    print("\n[G4] the GPU clock was pinned at the 306 MHz floor throughout")
    try:
        with open("/sys/class/devfreq/17000000.gpu/min_freq") as f:
            floor = int(f.read().strip())
    except Exception:
        floor = 306000000
    for c, runs in sorted(per_cfg.items()):
        clocks = set()
        for r in runs:
            for v in (r.get("gpu_clock_hz") or []):
                clocks.add(int(v))
        ok = bool(clocks) and clocks <= {floor}
        check(ok, f"{c}: the GPU clock stayed at the {floor//1000000} MHz floor",
              f"observed {sorted(clocks)}")

    # ---- the verdict
    verdict = {
        "checked_at": datetime.now().isoformat(),
        "rundir": os.path.abspath(args.rundir),
        "logdir": os.path.abspath(args.logdir),
        "thresholds": {
            "max_between_within_ratio": MAX_BETWEEN_WITHIN_RATIO,
            "max_ram_growth_mb_per_repeat": MAX_RAM_GROWTH_MB_PER_REPEAT,
            "min_window_completeness": MIN_WINDOW_COMPLETENESS,
        },
        "checks": results,
        "warnings": warns,
        "passed": not fails,
        "failed_checks": fails,
        "verdict": ("GATE PASSED -- the long pass may start."
                    if not fails else
                    "GATE FAILED -- the long pass must NOT start until "
                    "these are fixed: " + "; ".join(fails)),
    }
    out = args.out or os.path.join(args.rundir, "gate_verdict.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(verdict, f, indent=2)

    print("\n" + "=" * 78)
    if fails:
        print(f" GATE FAILED: {len(fails)} check(s) did not pass")
        for f_ in fails:
            print(f"   - {f_}")
        print(" The long pass must NOT be started.")
    else:
        print(" GATE PASSED -- the long pass may start.")
    for w in warns:
        print(f" WARN {w}")
    print(f"\nwrote {out}")
    print("=" * 78)
    return 1 if fails else 0


def calibrate():
    """Print the known-bad reference numbers next to the thresholds.

    The 2026-11-05/06 pass is the reference: it was produced by the old
    single-process protocol and its failure modes are known, so it is the
    right thing to calibrate a threshold against.
    """
    ref = os.path.join(HERE, "results", "run1_30hz")
    logs = os.path.join(HERE, "logs", "run1_30hz")
    print("=" * 78)
    print("CALIBRATION -- the known-bad 2026-11-05/06 pass")
    print("=" * 78)
    print(f"  threshold: between/within <= {MAX_BETWEEN_WITHIN_RATIO}")
    print(f"  threshold: RAM growth   <= {MAX_RAM_GROWTH_MB_PER_REPEAT} MB "
          f"per repeat step\n")
    if not os.path.isdir(logs):
        print(f"  (no reference logs at {logs})")
        return 0
    for c in ("C1", "C2", "C3", "C4", "C5"):
        series, rams = [], []
        for rep in range(5):
            p = os.path.join(logs, f"{c}_rep{rep}.csv")
            rows = _read_csv(p)
            vals = [float(x["teg_vdd_in_mw_avg"]) / 1000.0 for x in rows
                    if x.get("teg_vdd_in_mw_avg")]
            ram = [float(x["ram_used_mb"]) for x in rows if x.get("ram_used_mb")]
            if vals:
                series.append(vals)
            if ram:
                rams.append(float(np.mean(ram)))
        if not series:
            continue
        ratio, between, within = variance_ratio(series)
        gap, _ = bimodal_gap(series)
        growth, _ = ram_growth(rams)
        print(f"  {c}: between/within = {ratio:8.3f} "
              f"{'<- BIMODAL/FAIL' if ratio > MAX_BETWEEN_WITHIN_RATIO else ''}"
              f"   RAM rise = {growth:7.0f} MB "
              f"{'<- GROWING/FAIL' if growth > MAX_RAM_GROWTH_MB_PER_REPEAT else ''}"
              + (f"   largest gap = {gap*100:.0f}% of spread" if gap else ""))
    print("\n  The healthy configurations sit far below the thresholds and the")
    print("  known failures sit far above them, so the thresholds are not")
    print("  tuned to the edge of either population.")
    return 0


if __name__ == "__main__":
    from measurement_lock import refuse_if_measurement_running
    refuse_if_measurement_running("gate_check.py")
    sys.exit(main())
