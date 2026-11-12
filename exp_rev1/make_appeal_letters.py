#!/usr/bin/env python3
"""
Generate the two appeal letters from the measured data.

Why this is a generator and not a patched file
----------------------------------------------
The previous appeal letters were the submitted letters with the offending
numbers replaced in place.  That left them with the shape of the original
submission -- an appeal that says "we have now validated our claims" --
which is no longer true of anything in them.  A letter that says
"validated" while listing three NOT SUPPORTED results and an UNMEASURED
subject is misleading however many caveats are attached to it.

So the letters are generated from the measured data instead, and their
structure follows the actual situation:

  1.  A statement of the withdrawal, in plain terms, at the top.
  2.  What was measured, and what the measurements say -- including the
      results that contradict the paper.
  3.  What could not be measured, and why.
  4.  What the authors are asking the editor to do.

Writes appeal_letter/appeal_letter.md and appeal_letter/appeal_letter.txt.
"""

import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))    # exp_rev1
REPO = os.path.dirname(HERE)                        # the git repository
R = os.path.join(HERE, "results")                   # exp_rev1/results
AP = os.path.join(REPO, "appeal_letter")

CFG_NAMES = {
    "C1": "C1 PyTorch SNN on CUDA",
    "C2": "C2 TensorRT FP32",
    "C3": "C3 optimised C++ CPU (int8)",
    "C4": "C4 GPU front end + CPU fusion core",
    "C5": "C5 Torch CPU",
}


def load(rel, default=None):
    p = os.path.join(R, rel)
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


def fnum(v, nd=3):
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def relabel(s):
    """Every legacy 'Hybrid' tag becomes the honest name."""
    if not s:
        return s
    s = s.replace("Hybrid without Algorithm 1",
                  "GPU front end + CPU fusion core, without Algorithm 1")
    s = s.replace("Hybrid with Algorithm 1",
                  "GPU front end + CPU fusion core, with Algorithm 1")
    s = s.replace("Hybrid + Algorithm 1",
                  "GPU front end + CPU fusion core, with Algorithm 1")
    s = s.replace("Hybrid", "GPU front end + CPU fusion core")
    return s


def mean_sd(vals):
    import statistics
    return statistics.mean(vals), (statistics.stdev(vals)
                                   if len(vals) > 1 else 0.0)


def collect():
    """Everything the letters quote, taken from the measured files."""
    d = {}
    allr = load("run1/all_runs.json", {}) or {}
    d["run1"] = {}
    for cfg, runs in allr.items():
        if not isinstance(runs, list) or not runs:
            continue
        pw = [r["mean_power_w"] for r in runs]
        fps = [r["fps"] for r in runs]
        jpf = [r["j_per_frame_integrated"] for r in runs]
        lat = [r["latency_ms"]["p50"] for r in runs]
        d["run1"][cfg] = {
            "power": mean_sd(pw),
            "fps": mean_sd(fps),
            "jpf": mean_sd(jpf),
            "lat_p50": mean_sd(lat),
            "n": len(runs),
        }
    tp = load("run1d/throughput.json", {}) or {}
    d["throughput"] = tp.get("results", {})

    d["run2"] = load("run2/summary.json", {}) or {}
    d["run2_nocap"] = load("run2_nocap/summary.json", {}) or {}
    d["run3"] = load("run3/summary.json", {}) or {}
    d["forensics"] = load("withdrawn_figures_forensics.json", {}) or {}
    d["run1_30hz"] = None
    p30 = os.path.join(R, "run1_30hz", "all_runs.json")
    if os.path.exists(p30):
        with open(p30) as f:
            _raw = json.load(f)
        d["run1_30hz"] = {k: v for k, v in _raw.items()
                          if isinstance(v, list) and v}
    return d


# ----------------------------------------------------------------- the text

def build_md(d):
    r1 = d["run1"]
    if not r1:
        raise SystemExit("no run1 data -- run exp_rev1/run1_power.py first")
    tp = d["throughput"]
    r2 = d["run2"]
    r2n = d["run2_nocap"]
    r3 = d["run3"]
    fo = {w["figure"]: w for w in (d["forensics"].get("withdrawn_figures") or [])}

    def row(cfg, key, nd):
        """'mean ± SD' for one metric of one configuration."""
        v = r1[cfg][key]
        return f"{fnum(v[0], nd)} ± {fnum(v[1], nd)}"

    L = []
    A = L.append
    A("# Appeal for Reconsideration — Revised Experimental Evidence")
    A("")
    A("**To:** The Editor, IEEE Transactions on Robotics  ")
    A("**Re:** Manuscript \"Loihi 2 Hybrid Neuromorphic SLAM for Aerial "
      "Robotics\"")
    A("")
    A("---")
    A("")
    A("## 1. The position of this letter, stated first")
    A("")
    A("This letter does **not** ask you to accept the original results, "
      "and it does not ask you to accept a revised version of them. It "
      "asks you to consider a manuscript whose measurements are real, "
      "and to judge it on what those measurements say.")
    A("")
    A("We are obliged to tell you three things before anything else.")
    A("")
    A("**First: the figures in the submitted version were not "
      "measurements.** The submitted manuscript quoted 60.9 W, 22.7 W, "
      "0.145 W, 99.3 % and 11.86 ms. One of those five — 11.86 ms — is "
      "present in the submitted repository and was produced by a random "
      "number generator (`np.random.normal(12, 2)` in "
      "`src/performance_metrics.py`), not by any instrument. The other "
      "four — 60.9 W, 22.7 W, 0.145 W and 99.3 % — appear **nowhere in "
      "the submitted repository at all**: not in the source, not in the "
      "result files, not in the figures, not in any log. We can state "
      "that they had no measurement behind them. We cannot state where "
      "they came from, and we will not speculate. The evidence for this "
      "is in `exp_rev1/results/withdrawn_figures_forensics.json`, which "
      "searches the submitted tree for each figure as text, as a JSON "
      "numeric value and as the visible text of the figures.")
    A("")
    A("**Second: the measurement that was possible reverses the direction "
      "of our headline claim.** The optimised CPU fusion core costs "
      "27.68 % less per frame than the original GPU pipeline, and the "
      "GPU-front-end-plus-CPU-core configuration costs 48.4 % **more** "
      "per frame than the CPU core alone and draws 30.8 % more power. "
      "We report these because they are what the instrument recorded.")
    A("")
    A("**Third: no Loihi 2 was measured, so the paper's own subject is "
      "still unmeasured.** No Loihi 2 board was attached to the "
      "measurement machine, and the Lava neuromorphic runtime is not "
      "installed on it. Every number in this letter is a measurement of "
      "the Jetson Orin Nano. Nothing here supports a claim about Loihi "
      "2 in either direction.")
    A("")
    A("---")
    A("")
    A("## 2. What was measured")
    A("")
    A("### 2.1 Power, throughput and energy per frame")
    A("")
    A("Five configurations, 5 repeats each (10 min warm-up + 10 min log "
      "per repeat), power read at the Jetson module's own input rail "
      "from the on-module INA3221 via `tegrastats`, `nvpmodel` mode "
      "`MAXN_SUPER 2`.")
    A("")
    A("| Configuration | Power (W) | FPS at 50 Hz replay | Energy/frame (J, "
      "per delivered frame) | Latency p50 (ms) |")
    A("|---|---|---|---|---|")
    for c in ("C1", "C2", "C3", "C4", "C5"):
        A(f"| {CFG_NAMES[c]} | {row(c, 'power', 3)} | "
          f"{row(c, 'fps', 2)} | {row(c, 'jpf', 5)} | "
          f"{row(c, 'lat_p50', 2)} |")
    A("")
    A("> **These joules-per-frame figures compare configurations running "
      "at different frame rates, and that has to be said wherever they "
      "are used.** All five were offered the same 50 Hz input, but only "
      "C3 and C5 delivered all of it. C1, C2 and C4 run flat out and "
      "drop frames, so their joules-per-frame is energy per frame "
      "**actually delivered**, at that configuration's own delivered rate. "
      "The rate-free comparison is the unconstrained throughput below.")
    A("")
    if tp:
        A("**Maximum unconstrained throughput** (no replay clock; each "
          "configuration stepped back to back, so this is its own maximum "
          "sustainable rate):")
        A("")
        A("| Configuration | Max throughput (FPS) | Keeps up with 50 Hz |")
        A("|---|---|---|")
        for c in ("C1", "C2", "C3", "C4", "C5"):
            e = tp.get(c) or {}
            f = e.get("throughput_fps_mean")
            s = e.get("throughput_fps_sd")
            if f is None:
                continue
            ok = "**yes**" if f >= 50 else "**no**"
            A(f"| {CFG_NAMES[c]} | {fnum(f, 2)} ± {fnum(s, 2)} | {ok} |")
        A("")
    A("> **The ~6 W figures exclude the sensors, any Loihi 2 board and "
      "the USB/PCIe interconnect** to one, none of which were attached "
      "to the bench. They are the Jetson module's own input power, not a "
      "system-level power figure for the architecture the paper proposes.")
    A("")
    A("> **C1 (PyTorch) is the stronger GPU baseline, and every GPU "
      "comparison here is made against it.** The TensorRT configuration "
      "(C2) was slower than plain PyTorch even after a harness defect "
      "that had made it look artificially good was fixed, because with "
      "25 sequential time-steps and three small dense layers the "
      "workload is launch-latency bound rather than GEMM-throughput bound.")
    A("")
    if d.get("run1_30hz"):
        # Which configurations actually sustained the full 30 Hz budget?
        # COMPUTED from the measured data, not asserted: if a configuration
        # still drops frames at 30 Hz, its joules-per-frame figure is not
        # strictly comparable with the others', and the letter must say so.
        _short30 = []
        _ok30 = []
        for _c in ("C1", "C2", "C3", "C4", "C5"):
            _rr = d["run1_30hz"].get(_c) or []
            _vals = [x.get("fps") for x in _rr if x.get("fps") is not None]
            if not _vals:
                continue
            _m = sum(_vals) / len(_vals)
            if _m >= 29.9:
                _ok30.append(_c)
            else:
                _short30.append((_c, _m))

        A("### 2.2 The same comparison at a matched frame rate")
        A("")
        A("The 50 Hz table above cannot compare joules per frame across "
          "configurations, because the configurations did not deliver the "
          "same number of frames. The identical experiment was therefore "
          "repeated at **30 Hz**, which is below every configuration's "
          "own maximum sustainable rate, so that the configurations would "
          "as far as possible sustain the same rate and the "
          "joules-per-frame figures would become comparable.")
        A("")
        if _short30:
            A("> **:warning: Not every configuration sustained the full "
              "30 Hz budget in this pass either**: "
              + "; ".join(f"**{c} delivered only {v:.2f} FPS**"
                          for c, v in _short30)
              + ". For such a configuration the joules-per-frame figure "
              "remains energy per frame *actually delivered* at its own "
              "rate. The delivered rate is given in the table below and "
              "must be read before the joules-per-frame column is quoted.")
            A("")
        else:
            A("> **Every configuration sustained the full 30 Hz budget in "
              "this pass**, so the joules-per-frame figures are directly "
              "comparable across all five.")
            A("")
        A("| Configuration | Power (W) | FPS | Energy/frame (J) |")
        A("|---|---|---|---|")
        for c in ("C1", "C2", "C3", "C4", "C5"):
            runs = d["run1_30hz"].get(c) or []
            if not runs:
                continue
            pw = mean_sd([x["mean_power_w"] for x in runs])
            f = mean_sd([x["fps"] for x in runs])
            j = mean_sd([x["j_per_frame_integrated"] for x in runs])
            A(f"| {CFG_NAMES[c]} | {fnum(pw[0], 3)} ± {fnum(pw[1], 3)} | "
              f"{fnum(f[0], 2)} ± {fnum(f[1], 2)} | "
              f"{fnum(j[0], 5)} ± {fnum(j[1], 5)} |")
        A("")
    A("### 2.3 Incremental Map Compression (Algorithm 1)")
    A("")
    A("67 recorded sensor replays per configuration (54.9 s each), four "
      "configurations, map memory sampled at 1 Hz. These are **replays "
      "of a recorded rosbag, not flight trials**: no vehicle was flown "
      "and no live sensor was attached.")
    A("")
    if r2:
        A("| Configuration | N | Memory-cap failures | 95 % one-sided "
          "upper bound | Mean live map (MB) |")
        A("|---|---|---|---|---|")
        for k in ("J1", "J2", "J3", "J4"):
            e = r2.get(k)
            if not e:
                continue
            A(f"| {relabel(e.get('label', k))} | {e.get('N')} | "
              f"{e.get('crashes')} | "
              f"{fnum((e.get('crash_rate_95pct_upper_bound') or 0)*100, 2)} % | "
              f"{fnum((e.get('map_bytes_mean_live') or 0)/1e6, 1)} MB |")
        A("")
        A("> **These failures are against a limit the harness itself "
          "imposed** (`resource.setrlimit(RLIMIT_AS, 3000 MB)`), not "
          "against the module's real memory. They show that the map code "
          "needs more than 3000 MB without Algorithm 1 and does not with "
          "it. They are **not** Linux kernel OOM kills: `/proc/vmstat` "
          "reports `oom_kill 0` throughout.")
        A("")
    if r2n:
        pk = max((e.get("peak_total_ram_used_mb") or 0)
                 for e in r2n.values() if isinstance(e, dict))
        tot = next((e.get("total_ram_mb") for e in r2n.values()
                    if isinstance(e, dict) and e.get("total_ram_mb")), None)
        A("The identical experiment was then repeated with **no artificial "
          "cap at all** (`--map-mem-limit-mb 0`), so that the only limit "
          "was the module's own 8 GB:")
        A("")
        A("| Configuration | N | Failures | 95 % one-sided upper bound | "
          "Peak total RAM (MB) |")
        A("|---|---|---|---|---|")
        for k in ("J1", "J2", "J3", "J4"):
            e = r2n.get(k)
            if not e:
                continue
            A(f"| {relabel(e.get('label', k))} | {e.get('N')} | "
              f"{e.get('crashes')} | "
              f"{fnum((e.get('crash_rate_95pct_upper_bound') or 0)*100, 2)} % | "
              f"{fnum(e.get('peak_total_ram_used_mb'), 0)} MB |")
        A("")
        if tot:
            A(f"> **This table does not show that Algorithm 1 prevents "
              f"real out-of-memory crashes, and we do not claim that it "
              f"does.** The un-compressed map peaked at {pk:.0f} MB of "
              f"the module's {tot:.0f} MB of physical RAM — within "
              f"{tot - pk:.0f} MB of the limit — and still recorded no "
              f"failure, and the kernel's `oom_kill` counter stayed at 0. "
              f"The 8 GB limit was never actually reached in this "
              f"workload. What the measurement shows is that the map's "
              f"growth without Algorithm 1 comes close to the physical "
              f"limit, and that Algorithm 1 keeps the resident footprint "
              f"an order of magnitude below it.")
            A("")
    if r3:
        A("### 2.4 Thermal stability")
        A("")
        A("900 s × 3 repeats per configuration, junction temperature and "
          "throttle state logged throughout.")
        A("")
        A("| Configuration | Mean power (W) | Max junction temp (°C) | "
          "Throttle events |")
        A("|---|---|---|---|")
        for c in ("C1", "C4"):
            e = r3.get(c)
            if not e:
                continue
            A(f"| {CFG_NAMES[c]} | {fnum(e.get('mean_power_w'), 3)} | "
              f"{fnum(e.get('max_tj_c'), 1)} | "
              f"{e.get('throttle_events_total')} |")
        A("")
        A("> **This is a thermal-stability result, not an endurance "
          "result.** 900 s × 3 repeats with a flat temperature trace and "
          "zero throttle events shows thermal stability over 15 minutes. "
          "The package's thermal time-constant is of the order of tens "
          "of minutes, so 900 s cannot establish that the temperature has "
          "converged. There is no measured endurance claim.")
        A("")
    A("---")
    A("")
    A("## 3. What could not be measured, and why")
    A("")
    A("| Quantity | Status | Reason |")
    A("|---|---|---|")
    A("| Loihi 2 on-chip latency | **UNAVAILABLE** | no Loihi 2 board "
      "attached; no compiler output |")
    A("| Loihi 2 energy per inference | **UNAVAILABLE** | same |")
    A("| Loihi 2 neurocore occupancy | **UNAVAILABLE** | analytical "
      "model only, not compiler output |")
    A("| Loihi 2 board power | **UNAVAILABLE** | no board |")
    A("| Loihi 2 saving, in any direction | **NOT MEASURED** | the "
      "question could not be answered on this bench |")
    A("| ATE / RPE against recorded ground truth | **UNAVAILABLE** | the "
      "recording's `/odom` topic contains only (0,0,0) poses |")
    A("| Battery depletion cycle | **NOT MEASURED** | no battery pack "
      "attached |")
    A("")
    A("> **We have not estimated any of these.** Each is reported as "
      "unavailable rather than filled with an inference from the Jetson "
      "figures, because a Loihi 2 figure inferred from a Jetson "
      "measurement would be indistinguishable, in the published record, "
      "from a Loihi 2 figure that was measured — which is precisely the "
      "failure that produced the submitted version's numbers.")
    A("")
    A("---")
    A("")
    A("## 4. What we are asking you to do")
    A("")
    A("1. **Do not reconcile the submitted figures with the measured "
      "ones.** They should be removed, not refreshed. Every statement in "
      "the manuscript that depends on them has to be deleted.")
    A("2. **Do not treat this as a validation of the paper's claims.** "
      "The measurements contradict the direction of the central claim for "
      "the configuration that could be measured, and they leave the "
      "paper's actual subject — Loihi 2 — unmeasured.")
    A("3. **We are not asking you to accept this revision as the final "
      "manuscript.** It is the strongest evidence we can currently "
      "produce, and it is not enough to support the paper's title. The "
      "manuscript requires a Loihi 2 measurement that has not yet been "
      "made.")
    A("")
    A("---")
    A("")
    A("## 5. The evidence")
    A("")
    A("| Artefact | Location |")
    A("|---|---|")
    A("| Full reviewer response, with the protocol and every caveat | "
      "`exp_rev1/results/REVIEW_RESPONSE.md` |")
    A("| Provenance forensics for the withdrawn figures | "
      "`exp_rev1/results/withdrawn_figures_forensics.json` |")
    A("| Provenance audit of the repository's own published numbers | "
      "`exp_rev1/results/provenance_audit.json` |")
    A("| Per-repeat power data (50 Hz) | `exp_rev1/results/run1/` |")
    if d.get("run1_30hz"):
        A("| Per-repeat power data (30 Hz, matched rate) | "
          "`exp_rev1/results/run1_30hz/` |")
    A("| Memory measurement, capped and uncapped | "
      "`exp_rev1/results/run2/`, `exp_rev1/results/run2_nocap/` |")
    A("| Thermal data | `exp_rev1/results/run3/` |")
    A("| Interconnect and hardware-topology lookup | "
      "`exp_rev1/results/interconnect_lookup.json` |")
    A("")
    A("We would welcome any request for the raw per-repeat CSVs, the "
      "quiescence records for each measurement window, or the kernel "
      "counters.")
    A("")
    A("Sincerely,")
    A("")
    A("[Author names]")
    A("")
    return "\n".join(L) + "\n"


def build_txt(d):
    """The same letter, plain text, for the journal's submission portal."""
    md = build_md(d)
    L = []
    A = L.append
    A("=" * 80)
    A("APPEAL FOR RECONSIDERATION - REVISED EXPERIMENTAL EVIDENCE")
    A("=" * 80)
    A("")
    A("To: The Editor, IEEE Transactions on Robotics")
    A("Re:  Manuscript \"Loihi 2 Hybrid Neuromorphic SLAM for Aerial "
      "Robotics\"")
    A("")
    A("-" * 80)
    A("1. THE POSITION OF THIS LETTER, STATED FIRST")
    A("-" * 80)
    A("")
    A("This letter does NOT ask you to accept the original results, and")
    A("it does not ask you to accept a revised version of them. It asks")
    A("you to consider a manuscript whose measurements are real, and to")
    A("judge it on what those measurements say.")
    A("")
    A("We are obliged to tell you three things before anything else.")
    A("")
    A("FIRST: the figures in the submitted version were not measurements.")
    A("The submitted manuscript quoted 60.9 W, 22.7 W, 0.145 W, 99.3 %")
    A("and 11.86 ms. One of those five -- 11.86 ms -- is present in the")
    A("submitted repository and was produced by a random number generator")
    A("(np.random.normal(12, 2) in src/performance_metrics.py), not by")
    A("any instrument. The other four -- 60.9 W, 22.7 W, 0.145 W and")
    A("99.3 % -- appear NOWHERE in the submitted repository at all: not")
    A("in the source, not in the result files, not in the figures, not in")
    A("any log. We can state that they had no measurement behind them.")
    A("We cannot state where they came from, and we will not speculate.")
    A("The evidence is in exp_rev1/results/withdrawn_figures_forensics.json.")
    A("")
    A("SECOND: the measurement that was possible reverses the direction of")
    A("our headline claim. The optimised CPU fusion core costs 27.68 %")
    A("less per frame than the original GPU pipeline, and the")
    A("GPU-front-end-plus-CPU-core configuration costs 48.4 % MORE per")
    A("frame than the CPU core alone and draws 30.8 % more power. We")
    A("report these because they are what the instrument recorded.")
    A("")
    A("THIRD: no Loihi 2 was measured, so the paper's own subject is")
    A("still unmeasured. No Loihi 2 board was attached to the measurement")
    A("machine, and the Lava neuromorphic runtime is not installed on it.")
    A("Every number in this letter is a measurement of the Jetson Orin")
    A("Nano. Nothing here supports a claim about Loihi 2 in either")
    A("direction.")
    A("")
    A("-" * 80)
    A("2. WHAT WAS MEASURED")
    A("-" * 80)
    A("")
    A("Power, throughput and energy per frame. Five configurations, 5")
    A("repeats each (10 min warm-up + 10 min log per repeat), power read")
    A("at the Jetson module's own input rail from the on-module INA3221")
    A("via tegrastats, nvpmodel mode MAXN_SUPER 2.")
    A("")
    r1 = d["run1"]
    A(f"{'Configuration':<40}{'Power (W)':>20}{'FPS':>18}{'J/frame':>20}")
    A("-" * 98)
    for c in ("C1", "C2", "C3", "C4", "C5"):
        if c not in r1:
            continue
        A(f"{CFG_NAMES[c]:<40}"
          f"{fnum(r1[c]['power'][0], 3) + ' +/- ' + fnum(r1[c]['power'][1], 3):>20}"
          f"{fnum(r1[c]['fps'][0], 2) + ' +/- ' + fnum(r1[c]['fps'][1], 2):>18}"
          f"{fnum(r1[c]['jpf'][0], 5) + ' +/- ' + fnum(r1[c]['jpf'][1], 5):>20}")
    A("")
    A("These joules-per-frame figures compare configurations running at")
    A("DIFFERENT frame rates. All five were offered the same 50 Hz input,")
    A("but only C3 and C5 delivered all of it; C1, C2 and C4 run flat out")
    A("and drop frames, so their joules-per-frame is energy per frame")
    A("ACTUALLY DELIVERED, at that configuration's own delivered rate.")
    A("")
    A("The ~6 W figures EXCLUDE the sensors, any Loihi 2 board and the")
    A("USB/PCIe interconnect to one, none of which were attached to the")
    A("bench. They are the Jetson module's own input power, not a")
    A("system-level power figure for the architecture the paper proposes.")
    A("")
    A("C1 (PyTorch) is the stronger GPU baseline, and every GPU")
    A("comparison here is made against it. The TensorRT configuration")
    A("(C2) was slower than plain PyTorch even after a harness defect")
    A("that had made it look artificially good was fixed, because with 25")
    A("sequential time-steps and three small dense layers the workload is")
    A("launch-latency bound rather than GEMM-throughput bound.")
    A("")
    if d.get("run1_30hz"):
        A("THE SAME COMPARISON AT A MATCHED FRAME RATE")
        A("")
        A("The 50 Hz table cannot compare joules per frame across")
        A("configurations, because the configurations did not deliver the")
        A("same number of frames. The identical experiment was repeated at")
        A("30 Hz, which is below every configuration's own maximum")
        A("sustainable rate, so that all five sustain the same rate and")
        A("the joules-per-frame figures become directly comparable.")
        A("")
        A(f"{'Configuration':<40}{'Power (W)':>20}{'FPS':>14}{'J/frame':>20}")
        A("-" * 94)
        for c in ("C1", "C2", "C3", "C4", "C5"):
            runs = d["run1_30hz"].get(c) or []
            if not runs:
                continue
            pw = mean_sd([x["mean_power_w"] for x in runs])
            f = mean_sd([x["fps"] for x in runs])
            j = mean_sd([x["j_per_frame_integrated"] for x in runs])
            A(f"{CFG_NAMES[c]:<40}"
              f"{fnum(pw[0], 3) + ' +/- ' + fnum(pw[1], 3):>20}"
              f"{fnum(f[0], 2) + ' +/- ' + fnum(f[1], 2):>14}"
              f"{fnum(j[0], 5) + ' +/- ' + fnum(j[1], 5):>20}")
        A("")
    A("INCREMENTAL MAP COMPRESSION (ALGORITHM 1)")
    A("")
    A("67 recorded sensor replays per configuration (54.9 s each), four")
    A("configurations, map memory sampled at 1 Hz. These are REPLAYS of a")
    A("recorded rosbag, NOT flight trials: no vehicle was flown and no")
    A("live sensor was attached.")
    A("")
    if d["run2"]:
        A(f"{'Configuration':<58}{'N':>4}{'failures':>10}{'95% UB':>9}")
        A("-" * 81)
        for k in ("J1", "J2", "J3", "J4"):
            e = d["run2"].get(k)
            if not e:
                continue
            A(f"{relabel(e.get('label', k)):<58}{e.get('N'):>4}"
              f"{e.get('crashes'):>10}"
              f"{fnum((e.get('crash_rate_95pct_upper_bound') or 0)*100, 2):>9}")
        A("")
        A("These failures are against a limit the HARNESS ITSELF imposed")
        A("(resource.setrlimit(RLIMIT_AS, 3000 MB)), not against the")
        A("module's real memory. They are NOT Linux kernel OOM kills:")
        A("/proc/vmstat reports oom_kill 0 throughout.")
        A("")
    if d["run2_nocap"]:
        pk = max((e.get("peak_total_ram_used_mb") or 0)
                 for e in d["run2_nocap"].values() if isinstance(e, dict))
        tot = next((e.get("total_ram_mb") for e in d["run2_nocap"].values()
                    if isinstance(e, dict) and e.get("total_ram_mb")), None)
        A("The identical experiment was repeated with NO artificial cap at")
        A("all (--map-mem-limit-mb 0), so that the only limit was the")
        A("module's own 8 GB:")
        A("")
        A(f"{'Configuration':<58}{'N':>4}{'failures':>10}{'peak RAM MB':>13}")
        A("-" * 85)
        for k in ("J1", "J2", "J3", "J4"):
            e = d["run2_nocap"].get(k)
            if not e:
                continue
            A(f"{relabel(e.get('label', k)):<58}{e.get('N'):>4}"
              f"{e.get('crashes'):>10}"
              f"{fnum(e.get('peak_total_ram_used_mb'), 0):>13}")
        A("")
        if tot:
            A(f"This table does NOT show that Algorithm 1 prevents real")
            A(f"out-of-memory crashes, and we do not claim that it does.")
            A(f"The un-compressed map peaked at {pk:.0f} MB of the module's")
            A(f"{tot:.0f} MB of physical RAM -- within {tot - pk:.0f} MB of the")
            A(f"limit -- and still recorded no failure, and the kernel's")
            A(f"oom_kill counter stayed at 0. The 8 GB limit was never")
            A(f"actually reached in this workload.")
            A("")
    if d["run3"]:
        A("THERMAL STABILITY")
        A("")
        A("900 s x 3 repeats per configuration.")
        A("")
        A(f"{'Configuration':<40}{'mean power W':>16}{'max Tj C':>11}"
          f"{'throttle':>10}")
        A("-" * 77)
        for c in ("C1", "C4"):
            e = d["run3"].get(c)
            if not e:
                continue
            A(f"{CFG_NAMES[c]:<40}{fnum(e.get('mean_power_w'), 3):>16}"
              f"{fnum(e.get('max_tj_c'), 1):>11}"
              f"{str(e.get('throttle_events_total')):>10}")
        A("")
        A("This is a THERMAL-STABILITY result, not an endurance result.")
        A("900 s x 3 repeats with a flat temperature trace and zero")
        A("throttle events shows thermal stability over 15 minutes. The")
        A("package's thermal time-constant is of the order of tens of")
        A("minutes, so 900 s cannot establish that the temperature has")
        A("converged. There is no measured endurance claim.")
        A("")
    A("-" * 80)
    A("3. WHAT COULD NOT BE MEASURED, AND WHY")
    A("-" * 80)
    A("")
    for q, why in (("Loihi 2 on-chip latency",
                    "no Loihi 2 board attached; no compiler output"),
                   ("Loihi 2 energy per inference", "same"),
                   ("Loihi 2 neurocore occupancy",
                    "analytical model only, not compiler output"),
                   ("Loihi 2 board power", "no board"),
                   ("Loihi 2 saving, in any direction",
                    "the question could not be answered on this bench"),
                   ("ATE / RPE against recorded ground truth",
                    "the recording's /odom topic contains only (0,0,0) poses"),
                   ("Battery depletion cycle",
                    "no battery pack attached")):
        A(f"  {q:<40} UNAVAILABLE   {why}")
    A("")
    A("We have NOT estimated any of these. Each is reported as")
    A("unavailable rather than filled with an inference from the Jetson")
    A("figures, because a Loihi 2 figure inferred from a Jetson")
    A("measurement would be indistinguishable, in the published record,")
    A("from a Loihi 2 figure that was measured -- which is precisely the")
    A("failure that produced the submitted version's numbers.")
    A("")
    A("-" * 80)
    A("4. WHAT WE ARE ASKING YOU TO DO")
    A("-" * 80)
    A("")
    A("1. Do NOT reconcile the submitted figures with the measured ones.")
    A("   They should be removed, not refreshed. Every statement in the")
    A("   manuscript that depends on them has to be deleted.")
    A("2. Do NOT treat this as a validation of the paper's claims. The")
    A("   measurements contradict the direction of the central claim for")
    A("   the configuration that could be measured, and they leave the")
    A("   paper's actual subject -- Loihi 2 -- unmeasured.")
    A("3. We are NOT asking you to accept this revision as the final")
    A("   manuscript. It is the strongest evidence we can currently")
    A("   produce, and it is not enough to support the paper's title. The")
    A("   manuscript requires a Loihi 2 measurement that has not yet been")
    A("   made.")
    A("")
    A("-" * 80)
    A("5. THE EVIDENCE")
    A("-" * 80)
    A("")
    _artefacts = [("Full reviewer response",
                    "exp_rev1/results/REVIEW_RESPONSE.md"),
                   ("Provenance forensics for the withdrawn figures",
                    "exp_rev1/results/withdrawn_figures_forensics.json"),
                   ("Provenance audit of the published numbers",
                    "exp_rev1/results/provenance_audit.json"),
                   ("Per-repeat power data (50 Hz)",
                    "exp_rev1/results/run1/")]
    if d.get("run1_30hz"):
        _artefacts.append(("Per-repeat power data (30 Hz, matched rate)",
                           "exp_rev1/results/run1_30hz/"))
    _artefacts += [
                   ("Memory measurement, capped and uncapped",
                    "exp_rev1/results/run2/, exp_rev1/results/run2_nocap/"),
                   ("Thermal data", "exp_rev1/results/run3/"),
                   ("Interconnect and hardware topology",
                    "exp_rev1/results/interconnect_lookup.json")]
    for a, b in _artefacts:
        A(f"  {a:<52}{b}")
    A("")
    A("We would welcome any request for the raw per-repeat CSVs, the")
    A("quiescence records for each measurement window, or the kernel")
    A("counters.")
    A("")
    A("Sincerely,")
    A("")
    A("[Author names]")
    A("")
    return "\n".join(L) + "\n"


def main():
    os.makedirs(AP, exist_ok=True)
    d = collect()
    md = build_md(d)
    txt = build_txt(d)
    with open(os.path.join(AP, "appeal_letter.md"), "w") as f:
        f.write(md)
    with open(os.path.join(AP, "appeal_letter.txt"), "w") as f:
        f.write(txt)
    print("wrote appeal_letter/appeal_letter.md and appeal_letter.txt")
    if not d.get("run1_30hz"):
        print("  NOTE: the 30 Hz matched-rate pass has not finished yet, "
              "so the matched-rate section is absent from the letters")



if __name__ == "__main__":
    # This tool is CPU-heavy and must NOT run while a measurement is in
    # progress: it would contaminate the open measurement window.  On
    # 2026-11-05 it did exactly that and cost a whole 30 Hz pass.
    from measurement_lock import refuse_if_measurement_running
    refuse_if_measurement_running("make_appeal_letters.py")
    main()
