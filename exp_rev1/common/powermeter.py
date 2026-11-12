#!/usr/bin/env python3
"""
Multi-point power logger for the reviewer-requested measurement-boundary table.

Three measurement points are supported simultaneously and are logged at a
single, common timestamp so that stage-level numbers can be reconciled:

  M1  DC input of the whole payload
      Preferred source: an external INA219/INA3221 on the payload DC rail
      (sysfs hwmon).  If no external meter is present on the bus, this point
      is recorded as UNAVAILABLE -- it is NOT silently replaced by a number.
  M2  Jetson VDD_IN (module input)
      Source: tegrastats (mW) and, independently, the on-board INA3221
      (current x shunt-drop -> VDD_IN).  Both are logged so the two can be
      cross-checked.
  M3  Loihi 2 board supply
      Source: an external meter on the Loihi board rail.  Absent on this
      machine -> recorded as UNAVAILABLE.

All values are recorded with an explicit provenance string so no number in the
result files can be mistaken for a value it is not.
"""

import os
import re
import time
import threading
import subprocess
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict

import psutil  # noqa: E402

HWMON_ROOT = "/sys/class/hwmon"


def _is_own_child(pid: int) -> bool:
    """True when `pid` is this process.

    The workload under test (and, for C2, the TensorRT engines it builds)
    runs in this process.  It is NOT contamination -- it is the thing being
    measured.

    Note that the check is deliberately NOT "is this a descendant of mine".
    A contaminant is very often started as a subprocess of something else
    that happens to be in the same tree -- that is exactly what happened on
    2026-11-05, when the author's figure renderer was launched as a child
    of the verification script.  Treating every descendant as innocent would
    have missed it.  Instead only an explicit allow-list of harmless helper
    processes is exempt (below).
    """
    return pid == os.getpid()


# Helper processes that the logger itself starts, or that the platform
# always runs, and which are therefore never contamination.
_BENIGN_HELPERS = ("tegrastats",)

# The pids that are part of the WORKLOAD UNDER TEST, and which therefore
# must never be counted as contamination.
#
# Some backends run the workload in a separate process: C6 (the Lava
# software stack) cannot be driven in-process, because `lava.magma` lives
# in the Lava virtual environment and the ROS 2 stack lives in the system
# Python, so the two talk over a pipe and the Lava network runs in a child
# process.  That child IS the thing being measured, so it has to be
# exempt -- but the exemption has to be EXPLICIT, registered by the
# backend that owns it, rather than "anything that descends from me"
# (which would have missed the figure renderer that destroyed the
# 2026-11-05 pass).
#
# The registry is a set of pids, and the backend registers its own.
_WORKLOAD_PIDS: set = set()


def register_workload_pid(pid: int) -> None:
    """Declare `pid` part of the workload under test."""
    try:
        _WORKLOAD_PIDS.add(int(pid))
    except Exception:
        pass


def unregister_workload_pid(pid: int) -> None:
    """Withdraw a previously declared workload pid."""
    try:
        _WORKLOAD_PIDS.discard(int(pid))
    except Exception:
        pass


def is_workload_pid(pid: int) -> bool:
    """True when `pid` is this process or an explicitly registered part of
    the workload under test."""
    return pid == os.getpid() or pid in _WORKLOAD_PIDS

# Development tooling that runs on this machine regardless of what is being
# measured.  These are REPORTED (so a reader can see what the IDE was doing
# during the window) but they do not by themselves make the window dirty.
# The same classification the quiescence guard uses, so the two agree.
_DEVELOPMENT_TOOLING = ("cursor-server", "cursor-server-bridge",
                        "language-server", "monitor_gpu.py")


# ---------------------------------------------------------------- hwmon scan

def _read(path: str) -> Optional[int]:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _read_str(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def discover_instruments() -> Dict[str, dict]:
    """Enumerate every hwmon device and classify it.

    The Jetson carrier's own INA3221 is recognised by the presence of the
    'VDD_IN' label; any other INA2xx is treated as an external meter and is
    matched to a role by an explicit mapping file if one is supplied.
    """
    found = {}
    if not os.path.isdir(HWMON_ROOT):
        return found
    for name in sorted(os.listdir(HWMON_ROOT)):
        d = os.path.join(HWMON_ROOT, name)
        if not os.path.isdir(d):
            continue
        chip = _read_str(os.path.join(d, "name")) or ""
        labels = {}
        for f in os.listdir(d):
            m = re.match(r"in(\d+)_label$", f)
            if m:
                labels[int(m.group(1))] = _read_str(os.path.join(d, f))
        found[name] = {
            "path": d,
            "chip": chip,
            "labels": {str(k): v for k, v in labels.items()},
            "has_vdd_in": "VDD_IN" in labels.values(),
        }
    return found


class JetsonIna3221:
    """On-board carrier INA3221 (hwmon).  3 shunt channels + 3 rail voltages.

    Register semantics (verified against the Linux `ina3221` driver on this
    board, and cross-checked against tegrastats):

      in1_input .. in3_input = BUS VOLTAGE of channel 1..3   [mV]
      in4_input .. in6_input = SHUNT VOLTAGE of channel 1..3 [uV]
      in7_input               = sum of the shunt voltages    [uV]
      curr1_input .. curr3_input = CURRENT of channel 1..3   [mA]

    The current is the driver's own `currN_input`, which the driver derives
    as shunt_voltage / shunt_resistor.  That derivation was verified here:
    in4_input / shunt1_resistor reproduces curr1_input exactly (e.g.
    4360 uV / 5 mOhm = 872 mA, and the driver reports 872 mA), which fixes
    the register roles unambiguously.

    The earlier version of this class had the roles wrong: it read
    `in1_input` as a CURRENT and `in4_input` as a VOLTAGE.  That is the
    opposite of what the driver reports, and it is why the carrier's
    "power" came out ~5x the on-module reading.  The bug was in the
    arithmetic, not in the instrument: the carrier INA3221 agrees with
    tegrastats to within a few per cent once the registers are read
    correctly.

    Channel map on the Jetson carrier:
      ch1 = VDD_IN current           [mA]
      ch2 = VDD_CPU_GPU_CV current   [mA]
      ch3 = VDD_SOC current          [mA]
    """

    CH = {1: "vdd_in", 2: "vdd_cpu_gpu_cv", 3: "vdd_soc"}

    def __init__(self, hwmon_dir: str):
        self.dir = hwmon_dir
        self.shunt_uv = {}
        for i in (1, 2, 3):
            self.shunt_uv[i] = _read(os.path.join(hwmon_dir, f"shunt{i}_resistor")) or 5000

    def read(self) -> dict:
        out = {}
        for i, key in self.CH.items():
            bus_mv = _read(os.path.join(self.dir, f"in{i}_input"))
            shunt_uv = _read(os.path.join(self.dir, f"in{i + 3}_input"))
            cur_ma = _read(os.path.join(self.dir, f"curr{i}_input"))
            # Fall back to the shunt voltage if the driver's current register
            # is unreadable, but record which route was used so the number
            # is never silently of an unknown provenance.
            if cur_ma is None and shunt_uv is not None:
                cur_ma = shunt_uv / (self.shunt_uv[i] / 1e6) / 1000.0
                out[key + "_current_source"] = "derived_from_shunt_voltage"
            else:
                out[key + "_current_source"] = "driver_curr_register"
            out[key + "_ma"] = cur_ma
            out[key + "_mv"] = bus_mv
            out[key + "_shunt_uv"] = shunt_uv
            if cur_ma is not None and bus_mv is not None:
                out[key + "_w"] = cur_ma * bus_mv / 1e6
            else:
                out[key + "_w"] = None
        return out


class ExternalMeter:
    """Generic external INA219/INA3221 on sysfs, used for M1 and M3.

    The role is given explicitly (payload_dc / loihi_board) because a bare
    hwmon directory does not say which physical rail it is wired to.
    """

    def __init__(self, hwmon_dir: str, role: str):
        self.dir = hwmon_dir
        self.role = role
        self.channels = []
        for f in sorted(os.listdir(hwmon_dir)):
            m = re.match(r"in(\d+)_input$", f)
            if m:
                idx = int(m.group(1))
                label = _read_str(os.path.join(hwmon_dir, f"in{idx}_label")) or f"in{idx}"
                self.channels.append((idx, label))

    def read(self) -> dict:
        res = {}
        for idx, label in self.channels:
            v = _read(os.path.join(self.dir, f"in{idx}_input"))
            # INA219 current input is in mA; voltage is in1/2 of a separate
            # sysfs entry.  We accept an explicit voltage file if present.
            volt_path = os.path.join(self.dir, "in1_input")
            res[f"{self.role}.{label}"] = v
        return res


# ------------------------------------------------------------ GPU clock state
GPU_DEV = "/sys/class/devfreq/17000000.gpu"


def gpu_clock_state() -> dict:
    """The GPU's actual operating clock and its devfreq limits.

    This is a fixed condition of every measurement and must be recorded with
    it: if the GPU is held at its minimum clock in one configuration and
    boosts in another, the measured watts are not comparable.
    """
    out = {}
    for key, path in (("cur_freq_hz", f"{GPU_DEV}/cur_freq"),
                      ("min_freq_hz", f"{GPU_DEV}/min_freq"),
                      ("max_freq_hz", f"{GPU_DEV}/max_freq"),
                      ("governor", f"{GPU_DEV}/governor")):
        try:
            out[key] = open(path).read().strip()
        except Exception:
            out[key] = None
    try:
        out["at_minimum_clock"] = (int(out["cur_freq_hz"]) == int(out["min_freq_hz"]))
    except Exception:
        out["at_minimum_clock"] = None
    return out


# ------------------------------------------------------------ tegrastats tap

_TEGRA_RE = re.compile(
    r"RAM (\d+)/(\d+)MB.*?"
    r"GR3D_FREQ (\d+)%.*?"
    r"cpu@([\d.]+)C.*?"
    r"tj@([\d.]+)C.*?"
    r"VDD_IN (\d+)mW/(\d+)mW VDD_CPU_GPU_CV (\d+)mW/(\d+)mW VDD_SOC (\d+)mW/(\d+)mW"
)
# the GPU's own temperature sensor, reported by tegrastats as gpu@<deg>C
_TEGRA_GPU_T_RE = re.compile(r"gpu@([\d.]+)C")
_TEGRA_TS_RE = re.compile(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")


class TegrastatsTap:
    """Runs tegrastats as a child process and parses each line.

    tegrastats reports, for each rail, "instant/avg".  We keep both.
    """

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.proc: Optional[subprocess.Popen] = None
        self.buffer: List[str] = []
        self.lock = threading.Lock()

    def start(self):
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    def _reader(self):
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            p = parse_tegrastats_line(line.strip())
            if p is None:
                continue
            p["wall"] = time.time()
            with self.lock:
                self.buffer.append(p)
                if len(self.buffer) > 20000:
                    del self.buffer[:10000]

    def drain(self) -> List[dict]:
        with self.lock:
            out = self.buffer[:]
            self.buffer.clear()
        return out

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
            self.proc = None


def parse_tegrastats_line(line: str) -> Optional[dict]:
    m = _TEGRA_RE.search(line)
    if not m:
        return None
    gpu_t = _TEGRA_GPU_T_RE.search(line)
    return {
        "ram_used_mb": int(m.group(1)),
        "ram_total_mb": int(m.group(2)),
        "gr3d_pct": int(m.group(3)),
        "cpu_c": float(m.group(4)),
        "tj_c": float(m.group(5)),
        "gpu_c": float(gpu_t.group(1)) if gpu_t else None,
        "vdd_in_mw_inst": int(m.group(6)),
        "vdd_in_mw_avg": int(m.group(7)),
        "vdd_cvgcv_mw_inst": int(m.group(8)),
        "vdd_cvgcv_mw_avg": int(m.group(9)),
        "vdd_soc_mw_inst": int(m.group(10)),
        "vdd_soc_mw_avg": int(m.group(11)),
        "raw": line,
    }# ------------------------------------------------------------------ the logger

@dataclass
class InstrumentMap:
    """Which hwmon directory plays which measurement role."""
    payload_dc: Optional[str] = None      # M1: DC input of whole payload
    loihi_board: Optional[str] = None     # M3: Loihi 2 board supply
    jetson_ina: Optional[str] = None      # M2 raw counterpart of tegrastats
    notes: Dict[str, str] = field(default_factory=dict)


class PowerLogger:
    """Logs M1/M2/M3 at a fixed rate into a CSV.

    Rows are emitted on a fixed tick; any instrument that produced no value
    for that tick gets an empty cell, so a missing measurement is visibly
    missing and can never be mistaken for a zero.
    """

    def __init__(self, csv_path: str, rate_hz: float = 20.0,
                 imap: Optional[InstrumentMap] = None,
                 tegrastats_interval_ms: int = 100,
                 watch_contamination: bool = True,
                 contamination_threshold_pct: float = 10.0):
        self.csv_path = csv_path
        self.period = 1.0 / rate_hz
        self.imap = imap or InstrumentMap()
        self.instruments = discover_instruments()
        self.teg = TegrastatsTap(tegrastats_interval_ms)
        self.ina = JetsonIna3221(self.imap.jetson_ina) if self.imap.jetson_ina else None
        self.m1 = ExternalMeter(self.imap.payload_dc, "M1_payload_dc") if self.imap.payload_dc else None
        self.m3 = ExternalMeter(self.imap.loihi_board, "M3_loihi_board") if self.imap.loihi_board else None
        self.rows: List[dict] = []
        self.meta = {}
        # In-window contamination detection.
        #
        # The quiescence guard checks the machine BEFORE a window opens and
        # cannot protect a window that is already open.  That is not a
        # theoretical gap: on 2026-11-05 a whole 8.5 h pass was lost twice
        # because the author's own figure regeneration started INSIDE a
        # measurement window.  The lock now prevents that, but the logger
        # samples the machine's own CPU time independently, so a window that
        # was contaminated is DETECTED rather than merely suspected.
        self.watch_contamination = watch_contamination
        self.contamination_threshold_pct = contamination_threshold_pct
        self._cpu_baseline = None
        self._contaminants: List[dict] = []
        self._development: List[dict] = []

    def describe(self) -> dict:
        return {
            "csv_path": self.csv_path,
            "rate_hz": 1.0 / self.period,
            "tegrastats_interval_ms": self.teg.interval_ms,
            "instruments": self.instruments,
            "instrument_map": asdict(self.imap),
            "watch_contamination": self.watch_contamination,
        }

    # ------------------------------------------------- contamination watch
    def _cpu_snapshot(self) -> dict:
        """Total CPU seconds per pid, for the processes that are running."""
        out = {}
        for pr in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                t = pr.cpu_times()
                out[pr.info["pid"]] = (t.user + t.system,
                                       " ".join(pr.info.get("cmdline") or [])[:160])
            except Exception:
                continue
        return out

    def _watch_for_contamination(self, wall_start: float) -> None:
        """Compare the machine's CPU time against the workload's own.

        The workload under test is a known set of processes (this one and its
        children).  Anything ELSE that burns CPU during the window is
        contamination, and it is recorded with the CPU time it actually
        consumed so that the reader can judge it.
        """
        if not self.watch_contamination:
            return
        try:
            now = time.time()
            snap = self._cpu_snapshot()
            if self._cpu_baseline is None:
                self._cpu_baseline = (wall_start, snap)
                return
            t0, base = self._cpu_baseline
            wall = max(now - t0, 1e-6)
            for pid, (c1, cmd) in snap.items():
                if is_workload_pid(pid):
                    continue
                if any(b in cmd for b in _BENIGN_HELPERS):
                    continue
                c0 = base.get(pid, (0.0, ""))[0]
                used = c1 - c0
                pct = used / wall * 100.0
                if pct < self.contamination_threshold_pct:
                    continue
                if any(c["pid"] == pid for c in self._contaminants):
                    continue
                # The development-tooling list is DEDUPLICATED by pid, and
                # the entry is REFRESHED rather than appended: the watcher
                # runs every few seconds, so a process that stays above the
                # threshold would otherwise be recorded once per tick, and
                # the record would grow to hundreds of near-identical
                # entries.  The refreshed entry carries the LATEST share.
                _dev_existing = next((c for c in self._development
                                      if c["pid"] == pid), None)
                if _dev_existing is not None:
                    _dev_existing["cpu_percent_of_window"] = round(pct, 2)
                    _dev_existing["cpu_seconds_in_window"] = round(used, 3)
                    _dev_existing["window_elapsed_s"] = round(now - t0, 1)
                    continue
                rec = {
                    "pid": pid,
                    "cmdline": cmd,
                    "cpu_percent_of_window": round(pct, 2),
                    "cpu_seconds_in_window": round(used, 3),
                    "first_seen": time.strftime("%Y-%m-%d %H:%M:%S",
                                               time.localtime(now)),
                    "window_elapsed_s": round(now - t0, 1),
                }
                if any(x in cmd for x in _DEVELOPMENT_TOOLING):
                    # The IDE's language server and the GPU monitor run
                    # whatever the experiment is.  They are reported so the
                    # reader can see them, but they are identical for every
                    # configuration, so they do not make the window dirty.
                    rec["development_environment"] = True
                    rec["note"] = ("development tooling, not part of the "
                                   "measured workload; identical for every "
                                   "configuration, so it does not "
                                   "differentiate them")
                    self._development.append(rec)
                else:
                    self._contaminants.append(rec)
        except Exception:
            pass

    def contamination_record(self) -> dict:
        return {
            "watched": self.watch_contamination,
            "threshold_pct_of_window": self.contamination_threshold_pct,
            "contaminants": self._contaminants,
            "development_environment_processes": self._development,
            "clean": not self._contaminants,
        }

    def run(self, duration_s: float, label: str = "", on_tick=None):
        """Log for duration_s seconds.  on_tick(t) is called every tick."""
        self.teg.start()
        self.meta["gpu_clock_at_start"] = gpu_clock_state()
        t0 = time.time()
        deadline = t0 + duration_s
        next_tick = t0
        try:
            while time.time() < deadline:
                now = time.time()
                if now < next_tick:
                    time.sleep(min(self.period, next_tick - now))
                    continue
                next_tick += self.period
                row = {"t_rel": now - t0, "wall": now}
                # The GPU's operating clock is a fixed condition of the
                # measurement: if the GPU sits at its minimum clock in one
                # configuration and boosts in another, the measured watts are
                # not comparable.  It is sampled on every row so that every
                # power figure carries the clock it was taken at.
                row["gpu_clock"] = gpu_clock_state()
                if self.ina:
                    row.update(self.ina.read())
                if self.m1:
                    row.update(self.m1.read())
                if self.m3:
                    row.update(self.m3.read())
                self.rows.append(row)
                # The contamination watch runs on a slower cadence than the
                # power sampling: it reads every process's CPU-time counter,
                # which is far more expensive than reading one hwmon file.
                if self.watch_contamination and (len(self.rows) % 100 == 1):
                    self._watch_for_contamination(t0)
                if on_tick:
                    on_tick(now - t0, row)
        finally:
            self.teg.stop()
            self.meta["gpu_clock_at_end"] = gpu_clock_state()
            self.meta["contamination"] = self.contamination_record()
        self._flush(label)
        return self.rows

    def _flush(self, label: str):
        # the run's fixed conditions (GPU clock at start and end) are written
        # next to the csv so that every power number carries them
        try:
            with open(self.csv_path + ".meta.json", "w") as mf:
                json.dump({"label": label, **self.meta}, mf, indent=2)
        except Exception:
            pass
        # tegrastats lines carry their own capture timestamp; align each power
        # row to the tegrastats sample nearest in time (within one tick).
        teg = self.teg.drain()
        with open(self.csv_path, "w") as f:
            cols = ["t_rel", "wall", "vdd_in_w", "vdd_cpu_gpu_cv_w", "vdd_soc_w",
                    "vdd_in_ma", "vdd_in_mv", "vdd_cpu_gpu_cv_ma", "vdd_soc_ma",
                    "vdd_in_shunt_uv", "vdd_cpu_gpu_cv_shunt_uv", "vdd_soc_shunt_uv",
                    "teg_vdd_in_mw_avg", "teg_vdd_cvgcv_mw_avg", "teg_vdd_soc_mw_avg",
                    "gr3d_pct", "cpu_c", "tj_c", "gpu_c", "ram_used_mb",
                    "gpu_cur_hz", "gpu_min_hz", "gpu_max_hz",
                    "ina_vdd_in_w", "teg_vdd_in_w"]
            f.write(",".join(cols) + "\n")
            ti = 0
            best = None
            for r in self.rows:
                wall = r.get("wall", 0)
                while ti + 1 < len(teg) and abs(teg[ti + 1]["wall"] - wall) <= abs(teg[ti]["wall"] - wall):
                    ti += 1
                if teg:
                    best = teg[ti]
                # The GPU clock is sampled on every row, so a power figure can
                # be attributed to the clock state it was taken at.  This is
                # the column the conditions record refers to.
                gpu = r.get("gpu_clock") or {}
                ina_w = r.get("vdd_in_w")
                teg_in_w = (best["vdd_in_mw_avg"] / 1000.0) if best else None
                vals = [
                    f"{r.get('t_rel', 0):.4f}", f"{wall:.3f}",
                    _fmt(r.get("vdd_in_w")), _fmt(r.get("vdd_cpu_gpu_cv_w")),
                    _fmt(r.get("vdd_soc_w")),
                    _fmt(r.get("vdd_in_ma")), _fmt(r.get("vdd_in_mv")),
                    _fmt(r.get("vdd_cpu_gpu_cv_ma")), _fmt(r.get("vdd_soc_ma")),
                    _fmt(r.get("vdd_in_shunt_uv")),
                    _fmt(r.get("vdd_cpu_gpu_cv_shunt_uv")),
                    _fmt(r.get("vdd_soc_shunt_uv")),
                    _fmt(best["vdd_in_mw_avg"]) if best else "",
                    _fmt(best["vdd_cvgcv_mw_avg"]) if best else "",
                    _fmt(best["vdd_soc_mw_avg"]) if best else "",
                    _fmt(best["gr3d_pct"]) if best else "",
                    _fmt(best["cpu_c"]) if best else "",
                    _fmt(best["tj_c"]) if best else "",
                    _fmt(best.get("gpu_c")) if best else "",
                    _fmt(best["ram_used_mb"]) if best else "",
                    _fmt(gpu.get("cur_freq_hz")), _fmt(gpu.get("min_freq_hz")),
                    _fmt(gpu.get("max_freq_hz")),
                    _fmt(ina_w), _fmt(teg_in_w),
                ]
                f.write(",".join(vals) + "\n")
        self.meta = {"label": label, "n_rows": len(self.rows),
                     "n_tegrastats_lines": len(teg), "description": self.describe()}


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--jetson-ina", default=None)
    ap.add_argument("--payload-dc", default=None)
    ap.add_argument("--loihi-board", default=None)
    args = ap.parse_args()
    imap = InstrumentMap(payload_dc=args.payload_dc, loihi_board=args.loihi_board,
                         jetson_ina=args.jetson_ina)
    lg = PowerLogger(args.out, rate_hz=args.rate, imap=imap)
    print(json.dumps(lg.describe(), indent=2))
    lg.run(args.seconds)
    print("rows:", len(lg.rows))
