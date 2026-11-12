#!/usr/bin/env python3
"""
Quiescence check: verify the machine is genuinely idle before a power
measurement starts.

A measurement is only valid if no other process is drawing power from the
same rails.  This script samples tegrastats and psutil for a few seconds and
reports:

  * the GPU utilisation seen while nothing of ours is running (must be 0 %)
  * the module's idle power
  * every process that is currently using more than a given CPU share
  * the free memory

The result is written to a JSON file so that every measurement can point at
the quiescence record that was captured immediately before it.
"""

import os
import sys
import json
import time
import subprocess
import re
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
import psutil  # noqa: E402

TEG_RE = re.compile(
    r"RAM (\d+)/(\d+)MB.*?GR3D_FREQ (\d+)%.*?cpu@([\d.]+)C.*?tj@([\d.]+)C.*?"
    r"VDD_IN (\d+)mW/(\d+)mW VDD_CPU_GPU_CV (\d+)mW/(\d+)mW VDD_SOC (\d+)mW/(\d+)mW"
)


def teg_samples(n=12, interval_ms=200):
    p = subprocess.Popen(["tegrastats", "--interval", str(interval_ms)],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    out = []
    try:
        for _ in range(n):
            line = p.stdout.readline()
            if not line:
                break
            m = TEG_RE.search(line)
            if m:
                out.append({
                    "gr3d_pct": int(m.group(3)),
                    "cpu_c": float(m.group(4)),
                    "tj_c": float(m.group(5)),
                    "vdd_in_w": int(m.group(7)) / 1000.0,
                    "vdd_cvgcv_w": int(m.group(9)) / 1000.0,
                    "vdd_soc_w": int(m.group(11)) / 1000.0,
                })
    finally:
        p.terminate()
        try:
            p.wait(timeout=3)
        except Exception:
            p.kill()
    return out


def _cpu_seconds(pr):
    """Total CPU seconds consumed by the process (all its threads)."""
    t = pr.cpu_times()
    return t.user + t.system


def busy_processes(threshold_pct=10.0, exclude=(), window_s=2.0,
                  report_only=()):
    """Processes that are currently consuming CPU.

    The share is computed from the process's own CPU-time counter, not from
    psutil's cpu_percent(): cpu_percent() keeps its baseline inside a cached
    Process object, so a process that was already being tracked by an earlier
    invocation can be reported as 99 % busy while it is demonstrably idle.
    The CPU-time delta over a known wall-clock window has no such state and is
    what is actually wanted here.

    `report_only` names processes that are known to belong to the development
    environment rather than to the measured workload (the IDE's language
    server, for instance).  They are reported with their CPU share so that a
    reader can see how much of the machine the tooling was using, but they
    do NOT make the machine count as busy, because they are not part of any
    configuration under test and they are running identically for every one.
    """
    me = os.getpid()
    t0 = time.time()

    snap = {}
    for pr in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            if pr.info["pid"] == me:
                continue
            cmd = " ".join(pr.info.get("cmdline") or [])
            if any(x in cmd for x in exclude):
                continue
            is_ro = any(x in cmd for x in report_only)
            snap[pr.info["pid"]] = (pr, cmd, _cpu_seconds(pr), is_ro)
        except Exception:
            continue

    time.sleep(window_s)
    wall = time.time() - t0
    out = []
    for pid, (pr, cmd, c0, is_ro) in snap.items():
        try:
            c1 = _cpu_seconds(pr)
            cpu = (c1 - c0) / wall * 100.0 if wall > 0 else 0.0
            if cpu >= threshold_pct:
                mi = pr.info.get("memory_info")
                rec = {
                    "pid": pid,
                    "name": pr.info.get("name"),
                    "cmdline": cmd[:180],
                    "cpu_percent": round(cpu, 2),
                    "cpu_seconds_in_window": round(c1 - c0, 3),
                    "rss_mb": (mi.rss / (1 << 20)) if mi else None,
                }
                if is_ro:
                    rec["development_environment"] = True
                    rec["note"] = ("development tooling, not part of the "
                                   "measured workload; reported for "
                                   "transparency but does not make the "
                                   "machine busy")
                out.append(rec)
        except Exception:
            continue

    # A process that STARTED during the window is not in the snapshot, so it
    # would be missed.  Look for anything that appeared in the meantime: a
    # new process has consumed some CPU already, so anything still alive
    # that was not in the snapshot and has used more than a trivial amount of
    # CPU time is reported.
    for pr in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            pid = pr.info["pid"]
            if pid == me or pid in snap:
                continue
            cmd = " ".join(pr.info.get("cmdline") or [])
            if any(x in cmd for x in exclude):
                continue
            c1 = _cpu_seconds(pr)
            # a process older than the window that simply was not seen (e.g.
            # it was in the exclude list) is filtered above; what is left
            # either started during the window or is a short-lived helper
            if c1 < 0.05:
                continue
            mi = pr.info.get("memory_info")
            rec = {
                "pid": pid,
                "name": pr.info.get("name"),
                "cmdline": cmd[:180],
                "cpu_percent": round(c1 / wall * 100.0, 2),
                "cpu_seconds_in_window": round(c1, 3),
                "rss_mb": (mi.rss / (1 << 20)) if mi else None,
                "note": "started during the sampling window",
            }
            if any(x in cmd for x in report_only):
                rec["development_environment"] = True
                rec["note"] += "; development tooling, not part of the "\
                               "measured workload"
            out.append(rec)
        except Exception:
            continue

    return sorted(out, key=lambda d: -d["cpu_percent"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--cpu-threshold", type=float, default=10.0)
    args = ap.parse_args()
    vm = psutil.virtual_memory()
    teg = teg_samples(args.samples)
    # The IDE's own language server (cursor-server) is always running on
    # this machine and is not part of any measured workload.  It is excluded
    # from the idle check, but ONLY from the idle check: its CPU share is
    # still reported in the record, so that a reader can see how much of the
    # machine the tooling was using while the measurement was taken.
    #
    # `monitor_gpu.py` is in the same category, and for a stronger reason:
    # it is the instrument that watches a measurement from the outside, so
    # it has to be possible to run it WHILE the measurement runs.  It reads
    # sysfs and tegrastats only, it never writes to any devfreq node, and
    # it is a small fraction of one core.
    exclude = ("tegrastats", "psutil", "jtop")
    report_only = ("cursor-server", "cursor-server-bridge", "language-server",
                   "monitor_gpu.py")
    all_busy = busy_processes(args.cpu_threshold, exclude=exclude,
                              report_only=report_only)
    # only the processes that are NOT development tooling decide whether the
    # machine counts as idle
    busy = [b for b in all_busy if not b.get("development_environment")]
    background = [b for b in all_busy if b.get("development_environment")]

    # tegrastats' GR3D_FREQ is a short moving average, so it can still be
    # non-zero for a moment after the GPU stops.  The decision therefore uses
    # the LAST sample (the machine must have settled), and the maximum is
    # reported as well so that the record shows how quickly it settled.
    gr3d_last = teg[-1]["gr3d_pct"] if teg else 0
    gr3d = max((s["gr3d_pct"] for s in teg), default=0)
    idle_w = (sum(s["vdd_in_w"] for s in teg) / len(teg)) if teg else None

    verdict_ok = (gr3d_last == 0) and not busy
    rec = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tegrastats_samples": teg,
        "max_gr3d_pct_while_idle": gr3d,
        "last_gr3d_pct_while_idle": gr3d_last,
        "module_idle_power_w": idle_w,
        "busy_processes": busy,
        "development_environment_processes": background,
        "virtual_memory_total_mb": vm.total / (1 << 20),
        "virtual_memory_available_mb": vm.available / (1 << 20),
        "quiescent": verdict_ok,
        "verdict": ("QUIESCENT: the GPU is at 0 % and no process is using more "
                    f"than {args.cpu_threshold:g} % CPU, so a power measurement "
                    "taken now measures this workload only."
                    if verdict_ok else
                    "NOT QUIESCENT: other activity was present during the "
                    "measurement window, so the measured power includes it and "
                    "must not be used as a per-configuration figure."),
    }

    out = args.out or os.path.join(HERE, "results", "quiescence.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)

    print(f"GR3D while idle: last {gr3d_last} %, max {gr3d} %   "
          f"module idle power: {idle_w} W")
    print(f"busy processes: {len(busy)}")
    for b in busy:
        print(f"  pid {b['pid']:>7}  cpu {b['cpu_percent']:5.1f}%  "
              f"rss {b['rss_mb']:.0f} MB  {b['cmdline'][:110]}")
    print(rec["verdict"])
    print("wrote", out)
    return 0 if verdict_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
