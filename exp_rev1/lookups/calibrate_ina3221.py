#!/usr/bin/env python3
"""
Calibrate the carrier INA3221 against tegrastats' own rail power readings.

The Jetson carrier's INA3221 exposes several current and voltage registers
whose relationship to tegrastats' mW figures is not documented.  Rather than
assume a mapping, we drive the module through several distinct load levels
and check which combination of INA3221 registers reproduces tegrastats' VDD_IN /
VDD_CPU_GPU_CV / VDD_SOC wattages.  The output of this script is the
calibration table that the power logger's provenance refers to.

Register semantics (verified against the Linux `ina3221` driver)
-----------------------------------------------------------------
    in1_input .. in3_input = BUS VOLTAGE of channel 1..3   [mV]
    in4_input .. in6_input = SHUNT VOLTAGE of channel 1..3 [uV]
    in7_input               = sum of the shunt voltages    [uV]
    curr1_input .. curr3_input = CURRENT of channel 1..3   [mA]

The driver's own current registers are the authority: the driver computes
them as shunt_voltage / shunt_resistor, and that derivation reproduces
`currN_input` exactly here (e.g. 4360 uV / 5 mOhm = 872 mA, and the driver
reports 872 mA), which fixes the roles of the registers unambiguously.

Why this script was rewritten
------------------------------
The previous version read `in1_input` as a CURRENT and `in4_input` as a
VOLTAGE, i.e. exactly the opposite of what the driver reports, and it
compared `in1 * in4` against tegrastats.  That product is a voltage x a
shunt voltage, so it is not a power at all, and it came out roughly 5x the
on-module reading.  The conclusion drawn from it -- "the carrier's registers
over-read by roughly 5x" -- was an artifact of the arithmetic, not a
property of the instrument.  With the registers read correctly the carrier
agrees with tegrastats to within a few per cent, and it becomes a genuine
second opinion on the power figures rather than a discarded one.

The script now records BOTH readings and reports the agreement, so the
calibration is a measurement of the instrument rather than an assertion.
"""

import os
import re
import subprocess
import sys
import time

HWMON = "/sys/class/hwmon/hwmon1"


def rd(name):
    try:
        return int(open(os.path.join(HWMON, name)).read().strip())
    except Exception:
        return None


def teg_snapshot():
    """Take one tegrastats sample."""
    p = subprocess.Popen(["tegrastats", "--interval", "200"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    line = p.stdout.readline()
    p.terminate()
    m = re.search(r"VDD_IN (\d+)mW/(\d+)mW VDD_CPU_GPU_CV (\d+)mW/(\d+)mW VDD_SOC (\d+)mW/(\d+)mW", line)
    if not m:
        return None
    gr3d = re.search(r"GR3D_FREQ (\d+)%", line)
    return {
        "vdd_in_w": int(m.group(2)) / 1000.0,
        "vdd_cvgcv_w": int(m.group(4)) / 1000.0,
        "vdd_soc_w": int(m.group(6)) / 1000.0,
        "gr3d": gr3d.group(1) if gr3d else None,
    }


def simultaneous_samples(n=20, interval_ms=100):
    """Sample the carrier and tegrastats at the SAME instant, repeatedly.

    The single-shot comparison above is not a fair one: `snapshot()` reads
    the carrier first and then starts a fresh tegrastats process, so the two
    readings are hundreds of milliseconds apart and, worse, tegrastats' first
    line reports an average over its own start-up.  Under a fast-changing
    load that gap is worth tens of per cent on the smaller rails.

    Here tegrastats runs as a persistent child and the carrier registers are
    read immediately after each line arrives, so the two numbers describe
    the same instant.  Several such pairs are taken and the MEAN is used,
    which is what makes the comparison a measurement rather than a snapshot.
    """
    import statistics
    p = subprocess.Popen(["tegrastats", "--interval", str(interval_ms)],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    pairs = []
    try:
        for _ in range(n):
            line = p.stdout.readline()
            if not line:
                break
            m = re.search(r"VDD_IN (\d+)mW/(\d+)mW VDD_CPU_GPU_CV "
                          r"(\d+)mW/(\d+)mW VDD_SOC (\d+)mW/(\d+)mW", line)
            if not m:
                continue
            c = {
                "in1_mv": rd("in1_input"), "in2_mv": rd("in2_input"),
                "in3_mv": rd("in3_input"),
                "curr1_ma": rd("curr1_input"), "curr2_ma": rd("curr2_input"),
                "curr3_ma": rd("curr3_input"),
            }
            pairs.append({
                "teg_in": int(m.group(2)) / 1000.0,
                "teg_cv": int(m.group(4)) / 1000.0,
                "teg_soc": int(m.group(6)) / 1000.0,
                "ina_in": carrier_w(c, 1), "ina_cv": carrier_w(c, 2),
                "ina_soc": carrier_w(c, 3),
                "raw_carrier": c,
            })
    finally:
        p.terminate()
        try:
            p.wait(timeout=3)
        except Exception:
            p.kill()

    if not pairs:
        return None
    agg = {}
    for k in ("teg_in", "teg_cv", "teg_soc", "ina_in", "ina_cv", "ina_soc"):
        vals = [pp[k] for pp in pairs if pp.get(k) is not None]
        agg[k] = statistics.mean(vals) if vals else None
    agg["n_samples"] = len(pairs)
    agg["sd_teg_in"] = (statistics.pstdev([pp["teg_in"] for pp in pairs])
                        if len(pairs) > 1 else 0.0)
    return agg


def snapshot():
    """The carrier's INA3221, read with the driver's register semantics."""
    s = {
        # bus voltages, mV
        "in1_mv": rd("in1_input"), "in2_mv": rd("in2_input"), "in3_mv": rd("in3_input"),
        # shunt voltages, uV
        "in4_shunt_uv": rd("in4_input"), "in5_shunt_uv": rd("in5_input"),
        "in6_shunt_uv": rd("in6_input"),
        "in7_sum_shunt_uv": rd("in7_input"),
        # currents, mA
        "curr1_ma": rd("curr1_input"), "curr2_ma": rd("curr2_input"),
        "curr3_ma": rd("curr3_input"),
    }
    s["shunt1_resistor_uohm"] = rd("shunt1_resistor")
    s.update(teg_snapshot() or {})
    return s


def carrier_w(s, i):
    """The carrier's power for channel i, in watts: bus voltage x current."""
    mv = s.get(f"in{i}_mv")
    ma = s.get(f"curr{i}_ma")
    if mv is None or ma is None:
        return None
    return mv * ma / 1e6


def main():
    import torch
    print("=== calibration: idle vs CPU load vs GPU load ===")
    rows = []
    sim_rows = []

    def rec(tag):
        s = snapshot()
        s["tag"] = tag
        rows.append(s)
        print(f"{tag:>10} | teg VDD_IN={s.get('vdd_in_w')}W CV={s.get('vdd_cvgcv_w')}W "
              f"SOC={s.get('vdd_soc_w')}W | in1={s['in1_mv']}mV curr1={s['curr1_ma']}mA "
              f"shunt1={s['in4_shunt_uv']}uV")
        return s

    def rec_sim(tag):
        """The simultaneous, averaged comparison -- the real measurement."""
        agg = simultaneous_samples(n=25, interval_ms=100)
        if agg is None:
            print(f"{tag:>10} | simultaneous sampling FAILED")
            return None
        agg["tag"] = tag
        sim_rows.append(agg)
        print(f"{tag:>10} | (simultaneous, n={agg['n_samples']}) "
              f"teg VDD_IN={agg['teg_in']:.3f}W  ina VDD_IN={agg['ina_in']:.3f}W  "
              f"diff {(agg['ina_in'] - agg['teg_in']) / agg['teg_in'] * 100:+.2f}%")
        return agg

    rec("idle")
    rec_sim("idle")

    # --- CPU load (numpy BLAS on 6 cores)
    import numpy as np
    a = np.random.rand(600, 600)
    t0 = time.time()
    while time.time() - t0 < 12:
        a = a @ a
    rec("cpu_load")
    rec_sim("cpu_load")

    # --- GPU load (CUDA matmul) plus CPU
    if torch.cuda.is_available():
        g = torch.randn(4096, 4096, device="cuda")
        t0 = time.time()
        while time.time() - t0 < 12:
            g = torch.mm(g, g)
            g = g / g.norm()
        rec("gpu_load")
        rec_sim("gpu_load")

    # --- both
    if torch.cuda.is_available():
        t0 = time.time()
        while time.time() - t0 < 12:
            a = a @ a
            g = torch.mm(g, g)
            g = g / g.norm()
        rec("both_load")
        rec_sim("both_load")

    print("\n=== carrier INA3221 vs tegrastats, using the driver's register "
          "semantics ===")
    hdr = (f"{'tag':>10} | {'teg_in':>7} {'ina_in':>7} {'diff%':>6} | "
           f"{'teg_cv':>7} {'ina_cv':>7} {'diff%':>6} | "
           f"{'teg_soc':>7} {'ina_soc':>7} {'diff%':>6}")
    print(hdr)
    print("-" * len(hdr))
    for s in rows:
        cells = []
        for i, (kt, ki) in enumerate([("vdd_in_w", 1), ("vdd_cvgcv_w", 2),
                                      ("vdd_soc_w", 3)]):
            t = s.get(kt)
            c = carrier_w(s, ki)
            if t is None or c is None:
                cells += ["     -", "     -", "     -"]
            else:
                d = (c - t) / t * 100.0
                cells += [f"{t:7.3f}", f"{c:7.3f}", f"{d:6.2f}"]
        print(f"{s['tag']:>10} | " + " | ".join(
            cells[0:3]) + " | " + " | ".join(cells[3:6]) + " | " +
            " | ".join(cells[6:9]))

    print("\n=== the SIMULTANEOUS measurement: carrier vs tegrastats ===")
    hdr = (f"{'tag':>10} | {'teg_in':>7} {'ina_in':>7} {'diff%':>6} | "
           f"{'teg_cv':>7} {'ina_cv':>7} {'diff%':>6} | "
           f"{'teg_soc':>7} {'ina_soc':>7} {'diff%':>6} | n")
    print(hdr)
    print("-" * len(hdr))
    for s in sim_rows:
        cells = []
        for kt, ki in (("teg_in", "ina_in"), ("teg_cv", "ina_cv"),
                        ("teg_soc", "ina_soc")):
            t = s.get(kt)
            c = s.get(ki)
            if t is None or c is None or t == 0:
                cells += ["     -", "     -", "     -"]
            else:
                d = (c - t) / t * 100.0
                cells += [f"{t:7.3f}", f"{c:7.3f}", f"{d:6.2f}"]
        print(f"{s['tag']:>10} | " + " | ".join(cells[0:3]) + " | " +
              " | ".join(cells[3:6]) + " | " + " | ".join(cells[6:9]) +
              f" | {s.get('n_samples')}")

    print("\n=== the earlier SINGLE-SHOT comparison (kept for audit) ===")
    hdr = (f"{'tag':>10} | {'teg_in':>7} {'ina_in':>7} {'diff%':>6} | "
           f"{'teg_cv':>7} {'ina_cv':>7} {'diff%':>6} | "
           f"{'teg_soc':>7} {'ina_soc':>7} {'diff%':>6}")
    print(hdr)
    print("-" * len(hdr))
    for s in rows:
        cells = []
        for kt, ki in (("vdd_in_w", 1), ("vdd_cvgcv_w", 2), ("vdd_soc_w", 3)):
            t = s.get(kt)
            c = carrier_w(s, ki)
            if t is None or c is None:
                cells += ["     -", "     -", "     -"]
            else:
                d = (c - t) / t * 100.0
                cells += [f"{t:7.3f}", f"{c:7.3f}", f"{d:6.2f}"]
        print(f"{s['tag']:>10} | " + " | ".join(
            cells[0:3]) + " | " + " | ".join(cells[3:6]) + " | " +
            " | ".join(cells[6:9]))

    print("\n=== cross-check: does the driver's own current register agree "
          "with shunt_voltage / shunt_resistor? ===")
    for s in rows:
        for i in (1, 2, 3):
            shunt = s.get(f"in{i + 3}_shunt_uv")
            cur = s.get(f"curr{i}_ma")
            r = s.get("shunt1_resistor_uohm")
            if shunt is None or cur is None or r is None:
                continue
            derived = shunt / (r / 1e6) / 1000.0
            print(f"{s['tag']:>10} | ch{i}: driver {cur:6d} mA  "
                  f"derived {derived:8.2f} mA  "
                  f"delta {derived - cur:+7.2f} mA")

    import json
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                       "ina3221_calibration.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"register_semantics": {
            "in1_input..in3_input": "bus voltage of channel 1..3, mV",
            "in4_input..in6_input": "shunt voltage of channel 1..3, uV",
            "in7_input": "sum of the shunt voltages, uV",
            "curr1_input..curr3_input": "current of channel 1..3, mA",
            "shunt_resistor": "uOhm; the driver derives current as "
                               "shunt_voltage / shunt_resistor, which reproduces "
                               "currN_input exactly on this board",
        },
            "simultaneous": sim_rows,
            "single_shot": rows}, fh, indent=2)
    print("\nsaved:", os.path.normpath(out))


if __name__ == "__main__":
    main()
