#!/usr/bin/env python3
"""Recompute the stored "dropped" counts from the recorded elapsed time.

WHY THIS EXISTS
---------------
`common/workload.py` counted a frame as "dropped" every time it started
late, and then also advanced the deadline by one period, so the replay
clock stayed permanently behind and EVERY subsequent frame was counted as
a late drop as well.  One late frame therefore poisoned the count for the
whole window.

The clean 30 Hz pass recorded 16,294 "dropped" for C2 rep2, where the
number of genuinely missed deadlines was 13.

The counter has been fixed in `common/workload.py`.  This script repairs
the counts that were stored with the old counter, for the result files
that already exist, so that the published "dropped frames" figures are
the real number of missed deadlines.

THE REPAIR
----------
A frame is dropped when its deadline had already passed when the consumer
was ready to start it.  The number of slots the clock offered during the
logged window is `elapsed_s * rate_hz`, and the number of frames it
actually delivered is `frames`.  Every slot that was offered without a
frame being started on time is a missed deadline:

    dropped = max(0, int(elapsed_s * rate_hz) - frames)

This is the same identity the fixed counter satisfies by construction
(verified by exp_rev1/tests/test_replay_clock.py), so the repaired
values and the newly-measured values are computed the same way.

WHAT IS NOT AFFECTED
--------------------
Power, energy, joules per frame and FPS are all computed from the frames
actually delivered and the elapsed time.  They are NOT touched here.

The old count is preserved in the record as `dropped_old_counter` and the
reason is recorded in `dropped_note`, so the repair is auditable and a
reviewer can see what changed and why.

    python3 exp_rev1/repair_dropped_counts.py --rundir exp_rev1/results/run1_30hz
    python3 exp_rev1/repair_dropped_counts.py --rundir exp_rev1/results/run1_30hz --dry-run
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

NOTE = ("recomputed from elapsed_s and frames: the original counter "
        "double-counted a late frame for every frame that followed it "
        "(fixed in common/workload.py)")


def repair_run(run, rate_hz, dry=False):
    """Recompute one run record's dropped count.  Returns (old, new)."""
    elapsed = run.get("elapsed_s")
    frames = run.get("frames")
    if elapsed is None or frames is None:
        return None, None
    try:
        offered = int(float(elapsed) * float(rate_hz))
    except Exception:
        return None, None
    new = max(0, offered - int(frames))
    old = run.get("dropped")
    if dry:
        return old, new
    run["dropped_old_counter"] = old
    run["dropped"] = new
    run["dropped_note"] = NOTE
    return old, new


def iter_run_records(obj):
    """Yield every dict that looks like a run record, anywhere in obj."""
    if isinstance(obj, dict):
        if "frames" in obj and "elapsed_s" in obj:
            yield obj
        for v in obj.values():
            yield from iter_run_records(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_run_records(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True,
                    help="the results directory holding the run records")
    ap.add_argument("--rate-hz", type=float, default=None,
                    help="the replay rate; read from conditions.json if "
                         "not given")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change, write nothing")
    args = ap.parse_args()

    rundir = os.path.abspath(args.rundir)

    rate = args.rate_hz
    if rate is None:
        cond_path = os.path.join(rundir, "conditions.json")
        if os.path.exists(cond_path):
            with open(cond_path) as f:
                cond = json.load(f)
            rate = cond.get("rate_hz") or cond.get("rate")
        if rate is None:
            for fn in ("schedule.json",):
                p = os.path.join(rundir, fn)
                if os.path.exists(p):
                    with open(p) as f:
                        rate = json.load(f).get("rate_hz")
                    if rate:
                        break
    if not rate:
        print("could not determine the replay rate; pass --rate-hz",
              file=sys.stderr)
        return 2
    print(f"replay rate: {rate} Hz")
    print(f"rundir: {rundir}")

    # The per-repeat records are the source of truth; the per-configuration
    # and all_runs files are copies of them and are repaired too.
    targets = []
    reps_dir = os.path.join(rundir, "repeats")
    if os.path.isdir(reps_dir):
        for fn in sorted(os.listdir(reps_dir)):
            if fn.endswith(".json"):
                targets.append(os.path.join(reps_dir, fn))
    for fn in sorted(os.listdir(rundir)):
        if fn.endswith("_run.json") or fn == "all_runs.json":
            targets.append(os.path.join(rundir, fn))

    n_changed = 0
    n_files = 0
    for path in targets:
        with open(path) as f:
            data = json.load(f)
        changed = False
        for rec in iter_run_records(data):
            old, new = repair_run(rec, rate, dry=args.dry_run)
            if old is None:
                continue
            if old != new:
                changed = True
                n_changed += 1
                label = os.path.relpath(path, rundir)
                cfg = rec.get("config", "?")
                rep = rec.get("rep", "?")
                print(f"  {label} {cfg} rep{rep}: dropped {old} -> {new}")
        if changed and not args.dry_run:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        if changed:
            n_files += 1

    verb = "would change" if args.dry_run else "changed"
    print(f"\n{n_changed} run record(s) {verb} across {n_files} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
