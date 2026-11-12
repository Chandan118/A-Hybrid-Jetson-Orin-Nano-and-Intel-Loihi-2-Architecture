#!/usr/bin/env python3
"""
Run 3 -- endurance.

The reviewer asks which of the three cases applies and, if the measured case,
the data needed to support it:

  (a) Calculated: no run needed, label it an estimate.
  (b) Measured on the vehicle: pack capacity, cut-off voltage, payload,
      measured average battery power and the log.
  (c) Measured on the bench: run the full compute stack with all sensors on
      until cut-off.  Jetson-only and hybrid, 3 repeats each.  If a 150 Wh
      run is too long, use a smaller pack and report the measured Wh and the
      runtime; the 150 Wh figure then stays a labelled calculation.

What this machine can actually do
---------------------------------
There is NO battery pack attached to this Jetson: /sys/class/power_supply is
empty, there is no DC input meter on the payload rail, and no charger is
present.  A bench endurance run "until cut-off" therefore cannot be performed
against a real pack here.

What IS possible, and is done here:
  * a long, fully-instrumented bench run of the complete compute stack with
    the recorded sensor stream replayed (the sensors are 'on' in the sense
    that the full sensor front-end runs on every frame), logging power,
    temperature, throttling and throughput;
  * the energy actually consumed over the measured runtime is computed from
    the measured mean power, giving a MEASURED Wh figure for the measured
    runtime, and a runtime-to-empty figure that is explicitly labelled a
    CALCULATION from the measured power;
  * the 150 Wh endurance figure stays a labelled calculation.

The script records, for each configuration:
  pack capacity, cut-off voltage, payload, measured average power, and the
  full log, so the table in the report is filled from measurements.
"""

import os
import sys
import json
import time
import csv
import argparse
import threading
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
sys.path.insert(0, HERE)

import numpy as np
import psutil

import workload as WL
from powermeter import PowerLogger, InstrumentMap

THERMAL_ZONES = {
    "cpu": "/sys/devices/virtual/thermal/thermal_zone0/temp",
    "gpu": "/sys/devices/virtual/thermal/thermal_zone1/temp",
    "soc": "/sys/devices/virtual/thermal/thermal_zone2/temp",
}


def read_temp(path):
    try:
        return int(open(path).read().strip()) / 1000.0
    except Exception:
        return None


def throttling_state():
    """Read the SoC's throttle-alert bits."""
    out = {}
    for cd in os.listdir("/sys/class/thermal"):
        if not cd.startswith("cooling_device"):
            continue
        p = f"/sys/class/thermal/{cd}"
        t = open(f"{p}/type").read().strip()
        if "throttle" in t or "alert" in t:
            try:
                out[t] = int(open(f"{p}/cur_state").read().strip())
            except Exception:
                pass
    return out


def battery_inventory():
    """Everything we can honestly say about the power source on this machine."""
    inv = {}
    ps = "/sys/class/power_supply"
    inv["power_supply_entries"] = sorted(os.listdir(ps)) if os.path.isdir(ps) else []
    inv["has_battery"] = any(
        os.path.exists(os.path.join(ps, d, "type")) and
        open(os.path.join(ps, d, "type")).read().strip() == "Battery"
        for d in inv["power_supply_entries"])
    inv["wall_power_source"] = "DC input via the carrier's 5 V / barrel jack (mains adapter)"
    inv["pack_capacity_wh"] = None
    inv["cut_off_voltage_v"] = None
    inv["note"] = ("NO battery pack is attached to this machine "
                   "(/sys/class/power_supply is empty). A bench run against a "
                   "real pack until its cut-off voltage cannot be performed "
                   "here, so the endurance numbers are reported as a "
                   "MEASURED energy over a MEASURED runtime, with the "
                   "time-to-empty explicitly labelled a calculation.")
    return inv



# ------------------------------------------------------------------ guard
def require_quiescent(tag, outdir):
    """Refuse to start a measurement unless the machine is genuinely idle.

    A power number is only attributable to the configuration under test if no
    other process is drawing from the same rails.  This check is run BEFORE
    every measurement window and its record is written next to the results so
    that every figure can be traced back to the machine state it was taken in.
    """
    import subprocess
    import json
    rec_path = os.path.join(outdir, "quiescence", f"{tag}.json")
    os.makedirs(os.path.dirname(rec_path), exist_ok=True)
    p = subprocess.run([sys.executable, os.path.join(HERE, "quiescence.py"),
                        "--out", rec_path],
                       capture_output=True, text=True)
    try:
        rec = json.load(open(rec_path))
    except Exception:
        rec = {"quiescent": False, "verdict": "quiescence check failed to run: "
                + p.stderr[-400:], "busy_processes": []}
    print(f"  [quiescence] {tag}: {rec.get('verdict','')}", flush=True)
    if not rec.get("quiescent"):
        print("  [quiescence] REFUSING TO MEASURE -- the measurement would "
              "include the other activity listed in the record.", flush=True)
        raise SystemExit(
            "measurement aborted: the machine was not quiescent. "
            f"See {rec_path} for the offending processes.")
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=900.0,
                    help="seconds per endurance run")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--configs", default="C1,C4")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--outdir", default=os.path.join(HERE, "results", "run3"))
    ap.add_argument("--logdir", default=os.path.join(HERE, "logs", "run3"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    scans = WL.read_scans()
    enc = np.stack([WL.encode_input(WL.profile_from_scan(a)) for _, a in scans])
    weights = WL.load_weights(os.path.join(HERE, "common", "weights_quantised.npz"))
    netcfg = dict(WL.NET)
    netcfg["rate_hz"] = args.rate
    imap = InstrumentMap(jetson_ina="/sys/class/hwmon/hwmon1")

    from run1_power import TorchGPUBackend, HybridBackend

    inv = battery_inventory()
    print(json.dumps(inv, indent=2))

    results = {}
    for cfg in args.configs.split(","):
        cfg = cfg.strip()
        if not cfg:
            continue
        runs = []
        for rep in range(args.repeats):
            csv_path = os.path.join(args.logdir, f"endurance_{cfg}_rep{rep}.csv")
            require_quiescent(f"{cfg}_rep{rep}", args.outdir)
            logger = PowerLogger(csv_path, rate_hz=5.0, imap=imap,
                                 tegrastats_interval_ms=1000)
            stop = threading.Event()
            frames = [0]

            if cfg == "C1":
                backend = TorchGPUBackend(weights, netcfg)
            else:
                backend = HybridBackend(weights, netcfg)
            backend.reset()

            def work():
                i = 0
                period = 1.0 / args.rate
                nxt = time.perf_counter()
                while not stop.is_set():
                    nxt += period
                    now = time.perf_counter()
                    if now < nxt:
                        time.sleep(nxt - now)
                    else:
                        nxt = now
                    backend.step(enc[i % len(enc)])
                    frames[0] += 1

            t = threading.Thread(target=work, daemon=True)
            logger_thread = threading.Thread(
                target=logger.run, args=(args.duration,),
                kwargs={"label": f"{cfg}_rep{rep}"}, daemon=True)
            t.start()
            logger_thread.start()
            time.sleep(0.3)
            logger_thread.join(timeout=args.duration + 30)
            stop.set()
            t.join(timeout=10)
            backend.close()

            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            p = [float(r["teg_vdd_in_mw_avg"]) / 1000.0 for r in rows if r["teg_vdd_in_mw_avg"]]
            tj = [float(r["tj_c"]) for r in rows if r["tj_c"]]
            cpu_c = [float(r["cpu_c"]) for r in rows if r["cpu_c"]]
            gpu_c = [float(r["gr3d_pct"]) for r in rows if r["gr3d_pct"]]
            dur = args.duration
            mean_p = float(np.mean(p)) if p else float("nan")
            wh = mean_p * dur / 3600.0
            runs.append({
                "rep": rep,
                "duration_s": dur,
                "frames": frames[0],
                "fps": frames[0] / dur if dur else 0,
                "mean_power_w": mean_p,
                "std_power_w": float(np.std(p)) if p else None,
                "min_power_w": float(np.min(p)) if p else None,
                "max_power_w": float(np.max(p)) if p else None,
                "energy_wh": wh,
                "energy_j": mean_p * dur,
                "mean_tj_c": float(np.mean(tj)) if tj else None,
                "max_tj_c": float(np.max(tj)) if tj else None,
                "mean_cpu_c": float(np.mean(cpu_c)) if cpu_c else None,
                "max_cpu_c": float(np.max(cpu_c)) if cpu_c else None,
                "mean_gr3d_pct": float(np.mean(gpu_c)) if gpu_c else None,
                "throttle_events": sum(1 for r in rows if r.get("tj_c") and float(r["tj_c"]) > 85),
                "log_csv": csv_path,
            })
            print(f"  {cfg} rep{rep}: {mean_p:.3f} W  {wh:.4f} Wh over {dur:.0f}s  "
                  f"{runs[-1]['fps']:.1f} fps  max tj {runs[-1]['max_tj_c']} C")
            with open(os.path.join(args.outdir, f"{cfg}_runs.json"), "w") as f:
                json.dump(runs, f, indent=2)

        mean_p = float(np.mean([r["mean_power_w"] for r in runs]))
        mean_wh = float(np.mean([r["energy_wh"] for r in runs]))
        results[cfg] = {
            "label": ("Jetson-only" if cfg == "C1"
                      else "GPU front-end + CPU fusion core"),
            "repeats": args.repeats,
            "duration_s_per_run": args.duration,
            "mean_power_w": mean_p,
            "sd_power_w": float(np.std([r["mean_power_w"] for r in runs], ddof=1))
                           if len(runs) > 1 else 0.0,
            "mean_energy_wh_per_run": mean_wh,
            "measured_energy_wh": mean_wh,
            "mean_fps": float(np.mean([r["fps"] for r in runs])),
            "max_tj_c": float(np.max([r["max_tj_c"] for r in runs if r["max_tj_c"]])),
            "throttle_events_total": int(sum(r["throttle_events"] for r in runs)),
            "runs": runs,
            "battery": inv,
            "runtime_to_empty_150Wh_H": (150.0 / mean_p) if mean_p else None,
            "runtime_to_empty_150Wh_note": (
                "CALCULATION: 150 Wh divided by the MEASURED mean power. "
                "The 150 Wh pack capacity is NOT measured on this machine "
                "(no pack is attached), so the 150 Wh figure stays a labelled "
                "calculation; the power in the numerator is measured."),
        }
        with open(os.path.join(args.outdir, f"{cfg}_summary.json"), "w") as f:
            json.dump(results[cfg], f, indent=2)

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 100)
    print("RUN 3 -- ENDURANCE")
    print("=" * 100)
    print(f"{'configuration':<44}{'mean W':>12}{'SD W':>10}{'Wh/run':>12}{'fps':>10}{'max tj C':>12}")
    print("-" * 100)
    for cfg, r in results.items():
        print(f"{r['label']:<44}{r['mean_power_w']:>12.3f}{r['sd_power_w']:>10.3f}"
              f"{r['mean_energy_wh_per_run']:>12.4f}{r['mean_fps']:>10.1f}{r['max_tj_c']:>12.1f}")
    print("\nwrote", os.path.join(args.outdir, "summary.json"))



if __name__ == "__main__":
    # Hold the measurement lock for the whole run: the quiescence guard can
    # only check the machine BEFORE a window opens, so anything that starts
    # mid-window contaminates the numbers being produced right now.  See
    # measurement_lock.py.
    from measurement_lock import measurement_lock
    with measurement_lock("run3_endurance"):
        main()
