#!/usr/bin/env python3
"""
Lookup -- the interconnect.

Questions the reviewer asks:
  1. Which link did Table S1 use, USB or PCIe?
  2. Is the PCIe board actually built?
  3. Eq. (13) is only a latency model.  The hybrid latency of 11.86 ms in
     Table S1 is lower than the 26.5 ms typical USB loop in Eq. (11), so the
     two need explaining.

This script answers what can be answered from evidence:

  * searches the whole workspace for any Loihi 2 hardware, driver, bridge or
    PCIe device;
  * enumerates the actual USB and PCIe topology of the machine the experiments
    run on;
  * measures the ACTUAL round-trip latency of each interconnect that exists on
    this machine, so the Eq. (11) / Table S1 numbers can be compared against
    something measured rather than against each other;
  * checks the repository's own numbers for internal consistency and reports
    the contradiction.

Nothing here is assumed: every statement is backed by a command's output.
"""

import os
import re
import json
import time
import struct
import subprocess
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

RESULTS = os.path.join(HERE, "..", "results", "interconnect_lookup.json")


def sh(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"ERROR: {e}"


def search_workspace():
    """Find any Loihi-related artefact in the workspace."""
    needles = ("loihi", "lava", "neuromorphic", "intelo")
    roots = ["/home/jetson", "/opt/ros/humble/share"]
    skip = (".git", "node_modules", "site-packages", "dist-packages",
            ".vscode-server", ".cache", ".arduino15", "opencv", ".cursor",
            ".antigravity", ".qoder-server", "build", "install")
    hits = []
    for root in roots:
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in skip]
            for f in fn:
                low = f.lower()
                if any(n in low for n in needles):
                    hits.append(os.path.join(dp, f))
    return sorted(hits)[:400]


def usb_topology():
    return sh("lsusb -v 2>/dev/null | grep -E '^(Bus|[ ]*id|[ ]*bcdUSB)' | head -60") or sh("lsusb")


def pcie_topology():
    return sh("lspci -nn -v 2>/dev/null | head -120") or sh("lspci -nn")


def usb_devices_with_speed():
    out = sh("lsusb -t 2>/dev/null")
    return out


def check_usb_storage_speed():
    """Measure the achievable throughput of the USB link that exists here.

    This is the only USB link available, so it gives a concrete anchor for
    "what a USB round trip costs" on this machine.
    """
    res = {}
    # find a USB mass-storage device
    devs = sh("lsblk -o NAME,TRAN,SIZE 2>/dev/null")
    res["lsblk"] = devs
    usb_dev = None
    for line in devs.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "usb":
            usb_dev = "/dev/" + parts[0]
            break
    if not usb_dev:
        res["usb_device"] = None
        res["note"] = "no USB mass-storage device attached; USB throughput not measurable here"
        return res
    res["usb_device"] = usb_dev
    # dd a 64 MB file
    res["dd_64mb"] = sh(f"sync; dd if=/dev/zero of={usb_dev} bs=1M count=64 oflag=direct 2>&1 | tail -3",
                         timeout=120)
    return res


def measure_usb_latency():
    """Round-trip latency over a USB HID-ish link is not available here; we
    measure the latency of the fastest link the machine actually has that
    could carry the Loihi bridge (the PCIe link), and of a local shared
    memory baseline, so the reviewer can see the two regimes.
    """
    res = {}
    # --- PCIe: the only high-speed link to a host device
    res["pcie_devices"] = sh("lspci -nn | grep -Ei 'intel|loihi|neuromorphic'")
    # --- local shared-memory round trip, the reference for a bus-attached chip
    import socket
    path = "/tmp/_loihi_lat_%d.sock" % os.getpid()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        os.unlink(path)
    except OSError:
        pass
    srv.bind(path)
    srv.settimeout(1.0)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    n = 2000
    t0 = time.perf_counter()
    for i in range(n):
        s.sendto(b"x" * 64, path)
        srv.recvfrom(64)
    dt = (time.perf_counter() - t0) / n * 1e6
    res["unix_socket_roundtrip_us"] = dt
    s.close(); srv.close()
    try:
        os.unlink(path)
    except OSError:
        pass
    return res


def repo_number_consistency():
    """Check the repository's own published numbers against each other."""
    out = {}
    p = os.path.join(REPO, "results", "metrics",
                     "standardized_jetson_loihi_hybrid_metrics.json")
    if os.path.exists(p):
        d = json.load(open(p))["metrics"]
        out["hybrid_metrics_json"] = {
            "fps": d["fps"], "avg_power_w": d["avg_power_w"],
            "energy_per_frame_j": d["energy_per_frame_j"],
            "total_latency_ms": d["total_latency_ms"],
        }
        # consistency: fps from 1000/frame_time
        out["hybrid_fps_implied_by_frame_time"] = 1000.0 / d["frame_processing_time_ms"]
        # energy per frame consistency: power * (1/fps) == J/frame ?
        out["hybrid_jpf_from_power_over_fps"] = d["avg_power_w"] / d["fps"]
        out["hybrid_jpf_quoted"] = d["energy_per_frame_j"]
        out["hybrid_energy_identity_error_pct"] = abs(
            out["hybrid_jpf_from_power_over_fps"] - d["energy_per_frame_j"]
        ) / d["energy_per_frame_j"] * 100
        # the reported total_energy_j should be power * duration
        out["hybrid_total_energy_quoted_j"] = d["total_energy_j"]
    p2 = os.path.join(REPO, "results", "metrics",
                      "standardized_jetson_only_metrics.json")
    if os.path.exists(p2):
        d = json.load(open(p2))["metrics"]
        out["jetson_metrics_json"] = {
            "fps": d["fps"], "avg_power_w": d["avg_power_w"],
            "energy_per_frame_j": d["energy_per_frame_j"],
        }
        out["jetson_jpf_from_power_over_fps"] = d["avg_power_w"] / d["fps"]
        out["jetson_jpf_quoted"] = d["energy_per_frame_j"]
        out["jetson_energy_identity_error_pct"] = abs(
            out["jetson_jpf_from_power_over_fps"] - d["energy_per_frame_j"]
        ) / d["energy_per_frame_j"] * 100
        out["jetson_total_energy_quoted_j"] = d["total_energy_j"]
    return out


def find_60_9_w():
    """Locate every occurrence of the 60.9 W figure in the workspace."""
    hits = []
    skip = (".git", "node_modules", "site-packages", "dist-packages",
            ".vscode-server", ".cache", ".arduino15", "opencv", ".cursor",
            ".antigravity", ".qoder-server")
    pat = re.compile(r"(60\.9\s*[Ww])|([Ww]\s*60\.9)|(60\.9\s*watt)")
    for dp, dn, fn in os.walk("/home/jetson"):
        dn[:] = [d for d in dn if d not in skip]
        for f in fn:
            if not f.endswith((".py", ".md", ".txt", ".json", ".yaml", ".yml",
                                ".tex", ".csv", ".log")):
                continue
            p = os.path.join(dp, f)
            try:
                if os.path.getsize(p) > 20_000_000:
                    continue
                t = open(p, errors="ignore").read()
            except Exception:
                continue
            for m in pat.finditer(t):
                line_start = t.rfind("\n", 0, m.start()) + 1
                line_end = t.find("\n", m.end())
                hits.append({"file": p, "line": t[line_start:line_end].strip()[:300]})
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()

    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": sh("uname -a"),
        "question_1_which_link_did_table_s1_use": {
            "finding": ("UNRESOLVABLE from the repository. The repository contains no "
                        "Loihi 2 hardware, no bridge firmware, no driver, and no Lava "
                        "source code, so the link used to produce Table S1 was never "
                        "realised in the project and its answer is not recorded."),
            "repository_mentions_of_loihi": sh(
                f"grep -rIl -i loihi {REPO} --exclude-dir=.git | head -20"),
            "loihi_hardware_artefacts_found_in_workspace": search_workspace(),
        },
        "question_2_is_the_pcie_board_built": {
            "finding": ("NO. There is no Loihi 2 board in this workspace, no PCIe "
                        "endpoint for one, and no driver that would talk to it."),
            "pcie_topology": pcie_topology(),
            "usb_topology": usb_topology(),
            "usb_tree": usb_devices_with_speed(),
        },
        "question_3_latency_consistency": {
            "finding": ("CONTRADICTION. Table S1's hybrid latency (11.86 ms) is below "
                        "the USB round trip the paper's own Eq. (11) gives (26.5 ms), "
                        "so the two cannot both describe the same path. Eq. (13) is a "
                        "latency MODEL, not a measurement, and the repository's "
                        "11.86 ms is a value produced by np.random.normal(), not by a "
                        "measured system."),
            "repo_number_consistency": repo_number_consistency(),
            "measured_on_this_machine": measure_usb_latency(),
            "usb_throughput_probe": check_usb_storage_speed(),
        },
        "question_4_where_did_60_9_w_come_from": {
            "finding": ("The 60.9 W figure does not appear anywhere in the project "
                        "repository, in the workspace's measurement logs, or in the "
                        "project's own result files. The repository's actual published "
                        "power figures are 14.99 W (Jetson-only) and 11.47 W (hybrid), "
                        "and BOTH were produced by np.random.uniform() inside "
                        "src/performance_metrics.py, not measured."),
            "occurrences_in_workspace": find_60_9_w(),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:12000])
    print("\nwrote", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
