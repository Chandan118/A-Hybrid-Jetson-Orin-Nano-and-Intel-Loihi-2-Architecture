#!/usr/bin/env python3
"""Re-derive the Section 5.2 memory numbers from the RAW LOG FILES.

The summaries in results/run2/*_summary.json are the harness's own
aggregation.  Section 5.2 has to quote what is IN THE LOGS, so this
script re-derives every number directly from the per-run timeseries CSVs
and the per-run map-stat files, and then checks the two agree.

What is read
------------
  results/<pass>/J*_timeseries.csv   1 Hz samples of total and per-process
                                     memory for every replay of a pass
  results/<pass>/J*_mapstats.json    the map module's own state at the end
                                     of each replay
  results/<pass>/J*_summary.json     the harness aggregation (for cross-check)

Outputs results/memory_peak_from_logs.json.
"""
import csv
import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def scan_timeseries(path):
    """Peak of every memory column, and the run index it happened in."""
    peaks = {
        "total_used_mb": 0.0,
        "slam_rss_mb": 0.0,
        "lidar_rss_mb": 0.0,
        "map_rss_mb": 0.0,
        "ros2_rss_mb": 0.0,
    }
    n_runs = set()
    n_samples = 0
    with open(path) as f:
        for row in csv.DictReader(f):
            n_samples += 1
            try:
                n_runs.add(int(row["run"]))
            except (KeyError, TypeError, ValueError):
                pass
            for k in peaks:
                v = _f(row.get(k))
                if v is not None and v > peaks[k]:
                    peaks[k] = v
    peaks["n_distinct_runs"] = len(n_runs)
    peaks["n_samples"] = n_samples
    return peaks


def scan_mapstats(path):
    """The map module's final reported state for each replay."""
    out = []
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return out
    if isinstance(data, dict):
        data = [data]
    for d in data:
        if not isinstance(d, dict):
            continue
        out.append({
            "map_bytes": d.get("map_bytes"),
            "map_bytes_uncompressed": d.get("map_bytes_uncompressed"),
            "keyframes": d.get("keyframes"),
            "compression_events": d.get("compression_events"),
            "live_points": d.get("live_points"),
        })
    return out


def crashed_recordings(summary):
    """The harness numbering of the recordings that crashed.

    The harness increments `run_idx` and only THEN publishes it as
    state['run'], which is the `run` column of the timeseries CSV.  So the
    CSV label L belongs to harness recording L+1, and a crashed recording
    c appears in the CSV at label c-1.
    """
    return {c["run"] for c in (summary.get("crash_log") or [])}


def main():
    report = {}
    for pass_name, label in (("run2", "3000 MB RLIMIT_AS ceiling on the map process"),
                             ("run2_nocap", "no ceiling (--map-mem-limit-mb 0)")):
        pdir = os.path.join(RES, pass_name)
        if not os.path.isdir(pdir):
            continue
        print(f"\n=== {pass_name}: {label} ===")
        report[pass_name] = {"label": label, "configs": {}}
        for J in ("J1", "J2", "J3", "J4"):
            ts = os.path.join(pdir, f"{J}_timeseries.csv")
            summ = os.path.join(pdir, f"{J}_summary.json")
            if not os.path.exists(ts):
                continue
            pk = scan_timeseries(ts)
            ms = scan_mapstats(os.path.join(pdir, f"{J}_mapstats.json"))
            s = {}
            if os.path.exists(summ):
                s = json.load(open(summ))

            # Split the CSV by recording so the crashed recordings can be
            # separated from the healthy ones.
            crash_nums = crashed_recordings(s)
            crash_labels = {c - 1 for c in crash_nums}
            per_rec = defaultdict(list)
            with open(ts) as f:
                for row in csv.DictReader(f):
                    try:
                        lab = int(row["run"])
                        v = _f(row.get("total_used_mb"))
                    except (KeyError, TypeError, ValueError):
                        continue
                    if v is not None:
                        per_rec[lab].append(v)
            peak_healthy = max((max(v) for k, v in per_rec.items()
                                if k not in crash_labels), default=0.0)
            peak_crashed = max((max(v) for k, v in per_rec.items()
                                if k in crash_labels), default=0.0)
            peak_all = max((max(v) for v in per_rec.values()), default=0.0)

            total_ram = s.get("total_ram_mb")
            peak_total = pk["total_used_mb"]
            headroom = (total_ram - peak_total) if total_ram else None
            rec = {
                "label": s.get("label"),
                "n_replays": s.get("N"),
                "crashes": s.get("crashes"),
                "from_log": {
                    "peak_total_used_mb_all_recordings": peak_all,
                    "peak_total_used_mb_healthy_only": peak_healthy,
                    "peak_total_used_mb_crashed_recordings": peak_crashed,
                    "peak_map_rss_mb": pk["map_rss_mb"],
                    "peak_slam_rss_mb": pk["slam_rss_mb"],
                    "peak_lidar_rss_mb": pk["lidar_rss_mb"],
                    "peak_ros2_rss_mb": pk["ros2_rss_mb"],
                    "n_samples": pk["n_samples"],
                    "n_distinct_runs_in_log": pk["n_distinct_runs"],
                    "crashed_recordings_harness_numbering": sorted(crash_nums),
                },
                "total_ram_mb": total_ram,
                "headroom_mb": headroom,
                "headroom_percent": (headroom / total_ram * 100.0)
                                     if headroom is not None else None,
                "mapstats_records": len(ms),
                "map_bytes_live_last": ms[-1]["map_bytes"] if ms else None,
                "map_bytes_uncompressed_last":
                    ms[-1]["map_bytes_uncompressed"] if ms else None,
                "crosscheck_summary_peak_total_mb":
                    s.get("peak_total_ram_used_mb"),
                "crosscheck_summary_peak_components":
                    s.get("peak_by_component_rss_mb"),
                "summary_vs_log_note": (
                    "The harness summary updates its peak only at the "
                    "end of a SUCCESSFUL recording, so it skips the "
                    "samples of the recordings that crashed. The timeseries "
                    "CSV is the complete 1 Hz record and its maximum is "
                    "higher; the CSV value is the one quoted in the paper."),
            }
            report[pass_name]["configs"][J] = rec
            print(f"  {J} {s.get('label')}: crashes {s.get('crashes')}/{s.get('N')}")
            print(f"     log peak total (all recordings) {peak_all:9.1f} MB")
            print(f"        healthy only              {peak_healthy:9.1f} MB")
            print(f"        crashed recordings        {peak_crashed:9.1f} MB")
            print(f"     summary reported            "
                  f"{s.get('peak_total_ram_used_mb'):9.1f} MB  "
                  f"(skips crashed recordings)")
            print(f"     log peak map   {pk['map_rss_mb']:9.1f} MB   "
                  f"slam {pk['slam_rss_mb']:.1f}  lidar {pk['lidar_rss_mb']:.1f}  "
                  f"ros2 {pk['ros2_rss_mb']:.1f}")
            print(f"     headroom {headroom:.1f} MB "
                  f"({headroom / total_ram * 100:.2f} %)   "
                  f"logs cover {pk['n_distinct_runs']} recordings, "
                  f"{pk['n_samples']} samples")

    out = os.path.join(RES, "memory_peak_from_logs.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
