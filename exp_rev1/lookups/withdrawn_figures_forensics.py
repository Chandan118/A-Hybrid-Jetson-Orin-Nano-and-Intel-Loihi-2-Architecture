#!/usr/bin/env python3
"""
Provenance forensics for the figures quoted in the SUBMITTED manuscript.

Why this script exists
----------------------
Five figures were quoted in the submitted version of the manuscript:

    60.9 W   22.7 W   0.145 W   99.3 %   11.86 ms

The revision withdrew all five.  But "we withdrew them" is not an answer to
the question that actually has to be answered, which is: **where did they
come from?**  A withdrawal says what we are not claiming; a provenance
answer says what was claimed and how it was produced.

This script answers that question by searching the repository the way a
reviewer or an integrity officer would:

  1.  The ORIGINAL commit -- the tree that was submitted -- as the primary
      search space, because that is what the manuscript was written from.
  2.  Every file type: text, JSON, and the *visible text* of the SVG
      figures (a number printed on a chart is text, and searching only
      the source code would miss it).
  3.  JSON values numerically, not just as substrings, so that
      11.857080355410748 is matched by the claim "11.86 ms".
  4.  Against the generator lines that could have produced the number, so
      that a match can be followed all the way back to a line of code.

The output is results/withdrawn_figures_forensics.json.

The result is deliberately structured so that it distinguishes three
different situations, because they mean very different things:

  * FOUND_AND_EXPLAINED -- the number IS in the repository, and the
    generator that produced it is identified.  This is the ordinary
    "synthetic data" case.
  * FOUND_BUT_UNEXPLAINED -- the number IS in the repository, but no
    generator for it has been identified.  This is unexplained and must be
    reported as such.
  * NOT_FOUND -- the number appears NOWHERE in the submitted tree.  Its
    origin is not established by anything in this repository, and it must
    not be speculated about.
"""

import os
import re
import json
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))   # exp_rev1/lookups
EXP = os.path.dirname(HERE)                          # exp_rev1
REPO = os.path.dirname(EXP)                          # the git repository
R = os.path.join(EXP, "results")

# The five figures quoted in the submitted manuscript, with the tolerance
# used when matching a JSON number.  The tolerance is deliberately tight:
# a number that only matches at a loose tolerance is not the number.
WITHDRAWN = [
    {"figure": "60.9", "unit": "W", "kind": "power", "tol": 0.02,
     "claim": "system power"},
    {"figure": "22.7", "unit": "W", "kind": "power", "tol": 0.02,
     "claim": "system power"},
    {"figure": "0.145", "unit": "W", "kind": "power", "tol": 0.001,
     "claim": "Loihi 2 / neuromorphic core power"},
    {"figure": "99.3", "unit": "%", "kind": "percent", "tol": 0.05,
     "claim": "fusion-core energy saving"},
    {"figure": "11.86", "unit": "ms", "kind": "latency", "tol": 0.005,
     "claim": "hybrid latency, second row of the published table"},
]


def git(*args):
    r = subprocess.run(("git",) + args, cwd=REPO, capture_output=True,
                       text=True)
    return r.stdout


def original_commit():
    """The first commit -- the tree the manuscript was written from."""
    out = git("rev-list", "--all", "--reverse").split()
    return out[0] if out else None


def files_in(commit):
    return [f for f in git("ls-tree", "-r", "--name-only", commit).split("\n")
            if f.strip()]


def blob(commit, path):
    r = subprocess.run(("git", "show", f"{commit}:{path}"), cwd=REPO,
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def svg_text(svg):
    """The text a reader actually sees in an SVG: matplotlib stores its
    text elements as XML comments."""
    parts = re.findall(r"<!--\s*(.*?)\s*-->", svg, re.S)
    if parts:
        return "\n".join(parts)
    return "\n".join(re.findall(r">([^<>]{1,200})<", svg))


def numeric_values(obj, path=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(numeric_values(v, f"{path}.{k}" if path else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(numeric_values(v, f"{path}[{i}]"))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.append((path, float(obj)))
    return out


def find_generators(commit):
    """The lines of code in the SUBMITTED tree that could have produced a
    figure.

    The generator lines are read from the submitted commit, not from the
    working tree: the revision added a 20-line warning header to
    src/performance_metrics.py, so the line numbers differ between the two
    and a citation must point at the tree the manuscript was written from.
    """
    out = []
    for f in files_in(commit):
        if not (f.startswith("src/") and f.endswith(".py")):
            continue
        txt = blob(commit, f)
        if txt is None:
            continue
        for i, line in enumerate(txt.splitlines(), 1):
            if "np.random" in line:
                out.append({"file": f, "line_in_submitted_tree": i,
                            "text": line.strip()})
    return out


def generator_ranges(generators):
    """For each generator line: the range it draws from, and what KIND of
    quantity it produces.

    The kind matters.  A power figure must be matched against a generator
    that produces a POWER, and a latency against one that produces a TIME.
    Matching a 60.9 W figure against `np.random.normal(50, 10)` -- which
    produces a keyframe time in milliseconds -- would be a false positive,
    and this audit must not make one.
    """
    ranges = []
    for g in generators:
        t = g["text"]
        low = t.lower()
        if "record_power" in low or "power_w" in low or "system_power" in low:
            qkind = "power"
        elif ("frame_time" in low or "track_time" in low
              or "kf_time" in low or "latency" in low
              or "frame_processing_time" in low or "total_latency" in low):
            qkind = "latency"
        elif "energy" in low or "joule" in low:
            qkind = "energy"
        else:
            qkind = "unknown"
        mu = re.search(r"normal\(\s*([\d.]+)\s*,\s*([\d.]+)", t)
        lo = re.search(r"uniform\(\s*([\d.]+)\s*,\s*([\d.]+)", t)
        if mu:
            m, sd = float(mu.group(1)), float(mu.group(2))
            ranges.append({**g, "quantity": qkind, "dist": "normal",
                           "draws_from": [m - 3 * sd, m + 3 * sd]})
        if lo:
            a, b = float(lo.group(1)), float(lo.group(2))
            ranges.append({**g, "quantity": qkind, "dist": "uniform",
                           "draws_from": [min(a, b), max(a, b)]})
    return ranges


def main():
    commit = original_commit()
    files = files_in(commit)
    generators = find_generators(commit)
    ranges = generator_ranges(generators)

    for w in WITHDRAWN:
        fig = w["figure"]
        pat = re.compile(r"(?<![\d.])" + re.escape(fig) + r"(?![\d])")
        text_hits, json_hits, svg_hits = [], [], []
        for f in files:
            if f.endswith((".pdf", ".png")):
                continue
            txt = blob(commit, f)
            if txt is None:
                continue
            if pat.search(txt):
                text_hits.append(f)
            if f.endswith(".svg"):
                st = svg_text(txt)
                if pat.search(st):
                    svg_hits.append(f)
            if f.endswith(".json"):
                try:
                    d = json.loads(txt)
                except Exception:
                    continue
                for p, v in numeric_values(d):
                    if abs(v - float(fig)) <= w["tol"]:
                        json_hits.append({"file": f, "json_path": p,
                                          "value": v,
                                          "where": f"{f} :: {p}"})

        val = float(fig)
        cand = []
        for r in ranges:
            lo, hi = r["draws_from"]
            if not (lo <= val <= hi):
                continue
            # the generator must produce the SAME KIND of quantity
            if r["quantity"] != "unknown" and r["quantity"] != w["kind"]:
                continue
            cand.append({**r,
                         "note": "the figure lies inside the range this call "
                                 "draws from, and this call produces a "
                                 f"{r['quantity']}"})

        w["candidate_generators"] = cand
        w["occurrences_in_submitted_tree"] = {
            "text_files": sorted(set(text_hits)),
            "json_values": json_hits,
            "svg_visible_text": sorted(set(svg_hits)),
        }
        if (text_hits or json_hits or svg_hits) and cand:
            w["status"] = "FOUND_AND_EXPLAINED"
        elif text_hits or json_hits or svg_hits:
            w["status"] = "FOUND_BUT_UNEXPLAINED"
        else:
            w["status"] = "NOT_FOUND"

    out = {
        "purpose": ("Provenance forensics for the figures quoted in the "
                    "submitted manuscript.  Each figure is searched for in "
                    "the submitted tree -- as text, as a JSON numeric value, "
                    "and as the visible text of the SVG figures -- and then "
                    "matched against the generator lines that could have "
                    "produced it."),
        "scope_note": ("This audit establishes what the REPOSITORY contains. "
                       "It cannot establish where a number came from if the "
                       "number is not in the repository, and it does not "
                       "speculate about such numbers."),
        "searched_commit": commit,
        "files_in_submitted_tree": len(files),
        "withdrawn_figures": WITHDRAWN,
        "all_synthetic_generator_lines": generators,
    }
    os.makedirs(R, exist_ok=True)
    dst = os.path.join(R, "withdrawn_figures_forensics.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 78)
    print("PROVENANCE FORENSICS -- figures quoted in the submitted manuscript")
    print("=" * 78)
    print(f"submitted tree: commit {commit}, {len(files)} files\n")
    for w in WITHDRAWN:
        occ = w["occurrences_in_submitted_tree"]
        print(f"  {w['figure']} {w['unit']:<3} [{w['status']}]")
        if occ["text_files"]:
            print(f"        text in : {', '.join(occ['text_files'])}")
        for j in occ["json_values"]:
            print(f"        JSON    : {j['where']} = {j['value']}")
        if occ["svg_visible_text"]:
            print(f"        SVG text: {', '.join(occ['svg_visible_text'])}")
        if not (occ["text_files"] or occ["json_values"]
                or occ["svg_visible_text"]):
            print("        NOT PRESENT in the submitted tree")
        for c in w["candidate_generators"]:
            print(f"        generator: {c['file']}:"
                  f"{c['line_in_submitted_tree']}  {c['text']}")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
