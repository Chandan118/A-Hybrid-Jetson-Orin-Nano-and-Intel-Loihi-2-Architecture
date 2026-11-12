#!/usr/bin/env python3
"""
Live GPU / power monitor for the Jetson measurement bench.

Why this exists
---------------
Two separate reasons, and they matter for different measurements.

1.  **Attribution.**  The GPU's operating clock is a fixed condition of
    every power measurement.  If the GPU sits at its minimum clock (306 MHz
    on this board) in one configuration and boosts to 1020 MHz in another,
    the measured watts are not comparable and the comparison is invalid.
    `run1_power.py` samples the clock on every row of every power CSV (the
    `gpu_cur_hz` column) precisely so that this can be checked after the
    fact rather than assumed.

2.  **Contamination.**  A measurement is only attributable to the
    configuration under test if nothing else is drawing from the same rails.
    The quiescence guard (`quiescence.py`) checks the machine *before* each
    window; this monitor shows what is happening *during* one.  If this
    script shows a GPU utilisation that the workload under test cannot
    explain, the window is contaminated and the number must not be used.

Usage
-----
    # a live screen, refreshed every second
    python3 monitor_gpu.py

    # append samples to a CSV for later analysis, at 2 Hz, for 10 minutes
    python3 monitor_gpu.py --csv logs/gpu_monitor.csv --interval 0.5 --seconds 600

    # snapshot once, as JSON (used by the pipeline status script)
    python3 monitor_gpu.py --json

The monitor is a READ-ONLY observer.  It never writes to any devfreq node,
never changes the clock state and never touches the measurement lock, so it
can be run while a measurement is in progress.  It is deliberately excluded
from the quiescence guard's busy-process list for the same reason: it must
be possible to watch a measurement without perturbing it.  It reads sysfs
and tegrastats only, and it is a tiny fraction of one core.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))

from powermeter import gpu_clock_state  # noqa: E402

GPU_DEV = "/sys/class/devfreq/17000000.gpu"
HWMON = "/sys/class/hwmon/hwmon1"

TEG_RE = re.compile(
    r"RAM (\d+)/(\d+)MB.*?"
    r"GR3D_FREQ (\d+)%.*?"
    r"cpu@([\d.]+)C.*?"
    r"tj@([\d.]+)C.*?"
    r"VDD_IN (\d+)mW/(\d+)mW VDD_CPU_GPU_CV (\d+)mW/(\d+)mW VDD_SOC (\d+)mW/(\d+)mW"
)
GPU_T_RE = re.compile(r"gpu@([\d.]+)C")

CSV_COLS = [
    "wall", "t_rel",
    "gpu_cur_hz", "gpu_min_hz", "gpu_max_hz", "gpu_governor",
    "gr3d_pct", "gpu_c", "tj_c", "cpu_c",
    "teg_vdd_in_mw", "teg_vdd_cvgcv_mw", "teg_vdd_soc_mw",
    "ina_vdd_in_mw", "ina_vdd_in_ma", "ina_vdd_in_mv",
    "ram_used_mb", "ram_total_mb",
]


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def _read_int(path):
    v = _read(path)
    try:
        return int(v)
    except Exception:
        return None


def ina_vdd_in():
    """VDD_IN as the carrier's INA3221 sees it: bus voltage x current.

    `in1_input` is the BUS VOLTAGE in mV and `curr1_input` is the CURRENT in
    mA; the two multiply to watts.  (The earlier version of this code read
    `in1_input` as the current, which made the carrier appear to over-read
    tegrastats by ~5x.  The instrument was fine; the register roles were
    not.)
    """
    mv = _read_int(os.path.join(HWMON, "in1_input"))
    ma = _read_int(os.path.join(HWMON, "curr1_input"))
    w = (mv * ma / 1e6) if (mv is not None and ma is not None) else None
    return {"mv": mv, "ma": ma, "w": w}


class TegrastatsReader:
    """A tegrastats tap that yields one parsed sample per interval."""

    def __init__(self, interval_ms=500):
        self.interval_ms = interval_ms
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
            self.proc = None
        return False

    def read(self):
        if not self.proc or not self.proc.stdout:
            return None
        line = self.proc.stdout.readline()
        if not line:
            return None
        m = TEG_RE.search(line)
        if not m:
            return None
        g = GPU_T_RE.search(line)
        return {
            "ram_used_mb": int(m.group(1)),
            "ram_total_mb": int(m.group(2)),
            "gr3d_pct": int(m.group(3)),
            "cpu_c": float(m.group(4)),
            "tj_c": float(m.group(5)),
            "gpu_c": float(g.group(1)) if g else None,
            "teg_vdd_in_mw": int(m.group(7)),
            "teg_vdd_cvgcv_mw": int(m.group(9)),
            "teg_vdd_soc_mw": int(m.group(11)),
        }


def sample(teg):
    """One complete observation of the GPU and the board's power state."""
    gpu = gpu_clock_state()
    ina = ina_vdd_in()
    row = {
        "wall": time.time(),
        "gpu_cur_hz": gpu.get("cur_freq_hz"),
        "gpu_min_hz": gpu.get("min_freq_hz"),
        "gpu_max_hz": gpu.get("max_freq_hz"),
        "gpu_governor": gpu.get("governor"),
        "ina_vdd_in_ma": ina["ma"],
        "ina_vdd_in_mv": ina["mv"],
        "ina_vdd_in_mw": (ina["w"] * 1000.0) if ina["w"] is not None else None,
    }
    if teg is not None:
        row.update(teg)
    return row


def fmt_hz(v):
    if v in (None, "", "None"):
        return "  ?  "
    try:
        return f"{int(v) // 1000000:>4d}M"
    except Exception:
        return str(v)


def print_screen(row, prev=None):
    gr3d = row.get("gr3d_pct")
    cur = row.get("gpu_cur_hz")
    mn = row.get("gpu_min_hz")
    at_min = (str(cur) == str(mn)) and cur not in (None, "")
    warn = ""
    if at_min:
        warn = "  <-- GPU IS AT ITS MINIMUM CLOCK"
    if gr3d is not None and gr3d > 0:
        warn += "  [GPU BUSY]"
    print(f"  {time.strftime('%H:%M:%S')}  "
          f"GR3D {str(gr3d):>4}%  "
          f"gpu {fmt_hz(cur)} (min {fmt_hz(mn)})  "
          f"gpuT {row.get('gpu_c')}C  tj {row.get('tj_c')}C  "
          f"VDD_IN teg {row.get('teg_vdd_in_mw')}mW / ina "
          f"{row.get('ina_vdd_in_mw') and round(row['ina_vdd_in_mw'])}mW  "
          f"RAM {row.get('ram_used_mb')}MB{warn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="append samples to this CSV")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between samples (default 1.0)")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="how long to run; 0 means until interrupted")
    ap.add_argument("--json", action="store_true",
                    help="print one sample as JSON and exit")
    args = ap.parse_args()

    if args.json:
        with TegrastatsReader() as t:
            row = sample(t.read())
        print(json.dumps(row, indent=2))
        return 0

    fh = None
    writer = None
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        exists = os.path.exists(args.csv)
        fh = open(args.csv, "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        if not exists:
            writer.writeheader()

    t0 = time.time()
    deadline = t0 + args.seconds if args.seconds > 0 else None
    print(f"GPU monitor started at {time.strftime('%Y-%m-%d %H:%M:%S')}"
          f"  (interval {args.interval:g}s"
          + (f", until {time.strftime('%H:%M:%S', time.localtime(deadline))}"
             if deadline else ", until Ctrl-C") + ")")
    print("  The GPU devfreq node is root-owned, so the clock cannot be pinned")
    print("  on this bench.  'GPU IS AT ITS MINIMUM CLOCK' means the measured")
    print("  watts were taken with the GPU at 306 MHz, not at 1020 MHz.")
    try:
        with TegrastatsReader() as t:
            while True:
                row = sample(t.read())
                row["t_rel"] = row["wall"] - t0
                print_screen(row)
                if writer:
                    writer.writerow(row)
                    fh.flush()
                if deadline and time.time() >= deadline:
                    break
                time.sleep(max(0.0, args.interval - 0.5))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        if fh:
            fh.close()
    if args.csv:
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
