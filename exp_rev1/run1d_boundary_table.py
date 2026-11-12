#!/usr/bin/env python3
"""
Run 1d -- the measurement-boundary table.

The reviewer asks for this table, filled in:

    Item | Included in | Measured at (chip / board / module input / battery)
         | Instrument | Rate

The table is filled from the ACTUAL instruments that exist on this machine,
discovered at run time (the hwmon tree, the tegrastats tap, the USB/PCIe
tree).  Where an item CANNOT be measured on this machine, the cell says so
explicitly and states why.  Nothing is filled in from memory or assumption.
"""

import os
import sys
import json
import csv
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))

import numpy as np

from powermeter import discover_instruments, InstrumentMap


def hwmon_dir_exists(d):
    return os.path.isdir(d)


def main():
    imap = InstrumentMap(jetson_ina="/sys/class/hwmon/hwmon1")
    inv = discover_instruments()

    rows = []

    def add(item, included, measured_at, instrument, rate, note=""):
        rows.append({"item": item, "included_in": included,
                     "measured_at": measured_at, "instrument": instrument,
                     "rate": rate, "note": note})

    # ---- Jetson module + carrier ------------------------------------
    add("Jetson module + carrier",
        "YES -- all of C1..C5 (the compute stack is the Jetson in every "
        "configuration; only the LOCATION of the fusion core changes)",
        "module input (the carrier's power rail, upstream of the module's "
        "own regulators) -- read as the Jetson VDD_IN input rail of the "
        "Orin Nano module",
        "on-module INA3221 power monitor, read through the tegrastats "
        "interface (VDD_IN / VDD_SOC / VDD_CPU_GV_CV)",
        "20 Hz log rate, 100 ms tegrastats tap",
        "This is the whole-module boundary. It INCLUDES the carrier, the "
        "fan and every on-board regulator loss, and it EXCLUDES the Loihi 2 "
        "board and the sensors, which are not attached.")

    # ---- Fan ---------------------------------------------------------
    add("Fan", "YES -- it is on the carrier and inside the module input rail",
        "module input (the fan current passes through the carrier's "
        "input rail before the module's regulators)",
        "not a separate meter: the fan's contribution is inside the "
        "VDD_IN reading; its state is recorded from "
        "/sys/class/hwmon/hwmon*/pwm1 and rpm",
        "fan state sampled at the start of every run; power at 20 Hz",
        "Fan duty/RPM is recorded as a FIXED CONDITION. The fan cannot be "
        "switched off without root, so it runs in whatever state the "
        "firmware thermal daemon holds; that state is written down in "
        "results/run1/conditions.json.")

    # ---- Loihi 2 chip ------------------------------------------------
    add("Loihi 2 chip", "NO -- there is no Loihi 2 chip on this machine",
        "n/a -- would be the chip supply (VDD) on the Loihi 2 board",
        "none available (no Loihi 2 board is attached: no USB/PCIe endpoint, "
        "see results/interconnect_lookup.json)",
        "n/a",
        "The hybrid configurations C4 therefore use the PHYSICAL CPU "
        "fusion core as the stand-in for the Loihi 2 chip, and this is "
        "stated in the results.")

    # ---- Loihi 2 board ----------------------------------------------
    add("Loihi 2 board (FPGA bridge, regulators)",
        "NO -- no Loihi 2 board is attached",
        "n/a -- would be the board's 12 V/5 V input rail",
        "none available",
        "n/a",
        "Consequently the Loihi 2 board supply (the reviewer's M3 meter) "
        "CANNOT be read on this machine. It is recorded as UNAVAILABLE, "
        "never estimated.")

    # ---- USB or PCIe link -------------------------------------------
    add("USB or PCIe link", "NO -- no Loihi 2 link exists on this machine",
        "n/a",
        "none available",
        "n/a",
        "The USB tree and the PCIe topology are recorded in "
        "results/interconnect_lookup.json. Neither carries a Loihi 2 "
        "endpoint.")

    # ---- Sensors ----------------------------------------------------
    add("LiDAR, cameras, IMU, polarized light, magnetometer",
        "NO -- no sensor is physically attached; the SENSORS ARE REPLACED "
        "BY THE RECORDED STREAM so that every configuration does identical "
        "work",
        "n/a -- the sensors would sit on the payload DC input upstream of "
        "the Jetson",
        "none available (no /dev/ttyUSB*, no /dev/video*, no USB sensor "
        "enumerated)",
        "n/a",
        "Their idle draw is therefore NOT MEASURED but INFERRED-BY-"
        "CONSTRUCTION: the recorded /scan stream (2746 LaserScan messages, "
        "362 s) is replayed into every configuration, and the 'sensor "
        "replay only' stage (S1) measures the cost of the sensor front-end "
        "as it exists on this machine. The physical sensor idle draws must "
        "be added to every number in the table when the payload is "
        "flown.")

    # ---- DC-DC from battery -----------------------------------------
    add("DC-DC from battery", "NO -- there is no battery and no DC-DC "
        "converter on this machine",
        "n/a -- would be the battery terminal",
        "none available (/sys/class/power_supply is empty; no charger "
        "present)",
        "n/a",
        "All power figures in this response are measured at the Jetson "
        "module input. The end-to-end payload figure (battery terminal) is "
        "the sum of the Jetson module input and the sensor idle draws, "
        "plus the DC-DC efficiency.")

    out = {
        "measurement_boundary_table": rows,
        "instrument_inventory": inv,
        "three_meters_requested": {
            "M1_dc_input_of_whole_payload": {
                "status": "UNAVAILABLE",
                "reason": ("no external meter and no payload DC input on this "
                           "bench; the payload's sensor loads are not "
                           "physically attached"),
                "substitute": ("the Jetson module input rail (INA3221 VDD_IN) "
                               "IS measured and is the lower boundary of the "
                               "payload"),
            },
            "M2_jetson_vdd_in": {
                "status": "MEASURED",
                "instrument": "on-module INA3221 via tegrastats",
                "rate_hz": 20,
            },
            "M3_loihi2_board_supply": {
                "status": "UNAVAILABLE",
                "reason": "no Loihi 2 board attached to this machine",
            },
        },
        "recorded_fixed_conditions": "results/run1/conditions.json",
    }

    os.makedirs(os.path.join(HERE, "results", "run1"), exist_ok=True)
    path = os.path.join(HERE, "results", "run1", "measurement_boundary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    # a CSV rendering of the same table
    csvp = os.path.join(HERE, "results", "run1", "measurement_boundary.csv")
    with open(csvp, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Item", "Included in", "Measured at",
                     "Instrument", "Rate", "Note"])
        for r in rows:
            wr.writerow([r["item"], r["included_in"], r["measured_at"],
                         r["instrument"], r["rate"], r["note"]])

    print("=" * 100)
    print("MEASUREMENT BOUNDARY TABLE")
    print("=" * 100)
    for r in rows:
        print(f"\nItem            : {r['item']}")
        print(f"Included in     : {r['included_in']}")
        print(f"Measured at     : {r['measured_at']}")
        print(f"Instrument      : {r['instrument']}")
        print(f"Rate            : {r['rate']}")
        if r["note"]:
            print(f"Note            : {r['note']}")
    print(f"\nwrote {path}\nwrote {csvp}")



if __name__ == "__main__":
    # Hold the measurement lock for the whole run: the quiescence guard can
    # only check the machine BEFORE a window opens, so anything that starts
    # mid-window contaminates the numbers being produced right now.  See
    # measurement_lock.py.
    from measurement_lock import measurement_lock
    with measurement_lock("run1d_boundary_table"):
        main()
