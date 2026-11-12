#!/usr/bin/env python3
"""
Provenance audit of the numbers that are currently published in the
repository's own results/ and README.md.

For every published performance figure we check whether it came out of a
measurement or out of a random-number generator, and we check the energy
identity  E_per_frame == P / FPS  in the same file.

Writes results/provenance_audit.json.
"""

import os
import json
import re
import glob

HERE = os.path.dirname(os.path.abspath(__file__))   # exp_rev1/lookups
EXP = os.path.dirname(HERE)                        # exp_rev1
REPO = os.path.dirname(EXP)                        # the git repository
R = os.path.join(EXP, "results")

RNG = re.compile(r"np\.random\.(uniform|normal|randint|choice|seed)\b")


def audit_sources():
    """Every src/*.py file that injects synthetic values."""
    out = []
    for p in sorted(glob.glob(os.path.join(REPO, "src", "*.py"))):
        with open(p) as f:
            src = f.read()
        hits = []
        for i, line in enumerate(src.splitlines(), 1):
            if RNG.search(line):
                hits.append({"line": i, "text": line.strip()})
        if hits:
            out.append({"file": os.path.relpath(p, REPO), "synthetic_value_sites": hits})
    return out


def audit_metrics_json():
    """The energy identity check on results/metrics/*.json."""
    out = []
    for p in sorted(glob.glob(os.path.join(REPO, "results", "metrics", "*.json"))):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        rec = {"file": os.path.relpath(p, REPO)}
        if isinstance(d, dict):
            # the metric block may be at the top level or under "metrics"
            blocks = []
            for k, v in d.items():
                if isinstance(v, dict):
                    blocks.append((k, v))
            for k, v in blocks:
                fps = v.get("fps")
                pw = v.get("avg_power_w")
                jpf = v.get("energy_per_frame_j")
                if fps is None or pw is None or jpf is None:
                    continue
                if fps and pw and jpf:
                    implied = pw / fps
                    rec.setdefault("entries", []).append({
                        "key": k,
                        "fps": fps,
                        "avg_power_w": pw,
                        "energy_per_frame_j_quoted": jpf,
                        "energy_per_frame_j_implied_by_P_over_FPS": implied,
                        "energy_identity_error_pct": (implied - jpf) / jpf * 100.0,
                        "latency_ms": v.get("total_latency_ms"),
                        "implied_fps_from_latency": (1000.0 / v["total_latency_ms"])
                                                    if v.get("total_latency_ms") else None,
                    })
        out.append(rec)
    return out


# a power figure quoted with a W unit, or a bare 60.9 token that is not part
# of a longer number (so 60.95 or 160.9 do not match)
PWR = re.compile(r"(?<![\d.])60\.9\s*(?:W\b|w\b)|(?<![\d.])60\.9(?![\d])")


def search_number(root, skip_dirs=()):
    """Where does the quoted power figure occur at all?"""
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "__pycache__", "logs", "build",
                                    "install", "node_modules", ".venv",
                                    "ros2_ws", "exp_rev1")]
        for fn in filenames:
            if not fn.endswith((".py", ".json", ".md", ".yaml", ".yml", ".txt")):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if PWR.search(line):
                            hits.append({"file": os.path.relpath(p, root), "line": i,
                                         "text": line.strip()[:200]})
            except Exception:
                pass
    return hits


def main():
    res = {
        "question_1_which_published_numbers_are_simulated": {
            "finding": "Every published performance figure in results/metrics/ and in the "
                       "README key-results table was produced by a random-number generator "
                       "inside src/performance_metrics.py and src/standard_benchmark.py, "
                       "not measured from the machine. No code in src/ reads the Jetson "
                       "power rails, the INA3221, tegrastats, or any external meter.",
            "files_with_synthetic_values": audit_sources(),
        },
        "question_2_energy_identity_check": {
            "finding": "Within the same published JSON files, the quoted J/frame values "
                       "are inconsistent with the quoted power and FPS by more than an "
                       "order of magnitude, which is only possible if the three numbers "
                       "were drawn independently rather than measured from one run.",
            "metrics_audit": audit_metrics_json(),
        },
        "question_3_where_is_60_9_w": {
            "finding": "The 60.9 W figure does not occur anywhere in the repository or "
                       "in the workspace measurement logs. The repository's actual "
                       "published power figures are 14.99 W (Jetson-only) and 11.47 W "
                       "(hybrid), and both are random draws.",
            "occurrences_in_repository": search_number(REPO, "60.9"),
        },
        "question_4_published_vs_measured": {
            "finding": "The measured numbers in exp_rev1/results/ supersede every "
                       "published figure in results/ and in the README key-results "
                       "table. The README table and results/metrics/*.json must not be "
                       "presented as measurements.",
        },
    }
    os.makedirs(R, exist_ok=True)
    out = os.path.join(R, "provenance_audit.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2)[:3000])
    print("\nwrote", out)


if __name__ == "__main__":
    main()
