#!/usr/bin/env python3
"""Live status of the clean 30 Hz pass.

Read-only and very light: it reads a few small JSON/CSV files and prints
one screen.  It does NOT import torch, does not touch the GPU, and does
not run anything heavy, so it is safe to run while a measurement is open.

    python3 exp_rev1/pass_status.py
    python3 exp_rev1/pass_status.py --watch      # refresh every 30 s
"""

import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

RUNS = os.path.join(HERE, "results", "run1_30hz_clean")
LOGS = os.path.join(HERE, "logs", "run1_30hz_clean")
OUTER = os.path.join(LOGS, "outer.log")


def _load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _proc_elapsed():
    """Seconds since the orchestrator started, from its own start time."""
    try:
        with open(os.path.join(RUNS, "conditions.json")) as f:
            d = json.load(f)
        t0 = time.mktime(time.strptime(d["timestamp"][:19], "%Y-%m-%dT%H:%M:%S"))
        return time.time() - t0
    except Exception:
        return None


def _alive():
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().decode("utf-8", "replace")
        except Exception:
            continue
        if "run1_power.py" in cmd and "run1_30hz_clean" in cmd:
            return int(pid)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true",
                    help="refresh every 30 seconds")
    args = ap.parse_args()

    while True:
        os.system("clear")
        sched = _load(os.path.join(RUNS, "schedule.json")) or {}
        plan = sched.get("schedule") or []
        total = len(plan)

        reps_dir = os.path.join(RUNS, "repeats")
        done = {}
        if os.path.isdir(reps_dir):
            for fn in os.listdir(reps_dir):
                if fn.endswith(".json"):
                    d = _load(os.path.join(reps_dir, fn))
                    if d and "run" in d:
                        r = d["run"]
                        done[f"{r['config']}r{r['rep']}"] = r

        n = len(done)
        pid = _alive()
        elapsed = _proc_elapsed()

        print("=" * 72)
        print("CLEAN 30 Hz PASS -- live status")
        print("=" * 72)
        print(f"  running       : {'YES  pid ' + str(pid) if pid else 'NO -- STOPPED'}")
        if elapsed:
            print(f"  elapsed       : {elapsed/3600:.2f} h "
                  f"({elapsed/60:.0f} min)")
            # 25 repeats x 20 min of window, plus per-repeat overhead.
            if n and elapsed:
                eta = elapsed / n * (total - n)
                fin = time.time() + eta
                print(f"  progress      : {n}/{total} repeats "
                      f"({n*100//total}%)")
                print(f"  projected ETA : {eta/3600:.2f} h more, "
                      f"~{time.strftime('%H:%M', time.localtime(fin))}")
            else:
                print(f"  progress      : {n}/{total} repeats")

        # the repeat currently in flight
        cur = None
        for x in plan:
            key = f"{x['config']}r{x['rep']}"
            if key not in done:
                cur = key
                break

        print()
        print(f"  in flight     : {cur or 'none (finishing up)'}")
        print()
        print("  --- per repeat ------------------------------------------------")
        for x in plan:
            key = f"{x['config']}r{x['rep']}"
            r = done.get(key)
            if r:
                ram = (r.get("ram_used_mb") or {}).get("mean")
                print(f"    {key:<6} P={r['mean_power_w']:.4f} W  "
                      f"fps={r['fps']:.2f}  frames={r['frames']}  "
                      f"dropped={r['dropped']}  "
                      f"RAM={ram:.0f} MB")
            else:
                mark = "  <-- running" if key == cur else ""
                print(f"    {key:<6} pending{mark}")

        # the last few log lines, for anything alarming
        if os.path.exists(OUTER):
            try:
                with open(OUTER, errors="replace") as f:
                    tail = f.read().splitlines()[-6:]
            except Exception:
                tail = []
            print()
            print("  --- last log lines -------------------------------------------")
            for line in tail:
                print(f"    {line[:66]}")

        nvm = 0
        if os.path.exists(OUTER):
            try:
                with open(OUTER, errors="replace") as f:
                    nvm = f.read().count("NvMapMemAllocInternalTagged")
            except Exception:
                pass
        print()
        print(f"  NvMap errors so far : {nvm}"
              + ("   <-- investigate" if nvm else "   (clean)"))
        print("=" * 72)

        if not args.watch:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
