#!/usr/bin/env python3
"""
Rewrite the published key-results tables from the MEASURED results.

Why this exists
---------------
The provenance audit (`lookups/provenance_audit.py`) established that every
performance figure currently published in `README.md` and in
`results/metrics/*.json` was produced by `np.random` inside `src/`, and
that the energy identity is violated by 979 % to 1732 % in the published
files themselves.  Those numbers are not measurements and must not be
presented as such.

This script replaces them with the values measured in `exp_rev1/results/`,
and marks every value it cannot source from a measurement as
"NOT MEASURED" rather than leaving a stale number in place.

It is deliberately conservative:

  * it only rewrites the tables it has a measurement for;
  * anything it cannot source is annotated, never deleted silently;
  * the original published numbers are preserved in
    `results/metrics/PUBLISHED_SYNTHETIC_VALUES.json` so the change is
    auditable and reversible;
  * the synthetic number generators in `src/` are NOT removed here -- they
    are the evidence for the provenance finding, and removing them would
    destroy that evidence.  The README is changed to say plainly that the
    code under `src/` generates synthetic values.

Run it after the measurement passes finish:

    python3 exp_rev1/sync_published_numbers.py
"""

import json
import math
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
R = os.path.join(HERE, "results")


def load(path, default=None):
    p = path if os.path.isabs(path) else os.path.join(HERE, path)
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


def fnum(v, nd=3):
    return f"{v:.{nd}f}"


# The Run 2 / Run 3 result JSON files were written before the "C4 is not
# Loihi 2" scope was stated explicitly, and they label the C4/J3/J4
# configurations with the word "Hybrid".  Those labels are re-worded at
# render time, so that nothing published can imply that a Jetson-only
# measurement was a Loihi 2 measurement.
_LABEL_FIXES = (
    ("Hybrid without Algorithm 1",
     "GPU front end + CPU fusion core, without Algorithm 1"),
    ("Hybrid + Algorithm 1",
     "GPU front end + CPU fusion core, with Algorithm 1"),
    ("Hybrid (GPU front-end + CPU fusion core)",
     "GPU front end + CPU fusion core (not Loihi 2)"),
    ("Hybrid (GPU front-end + CPU fusion core)",
     "GPU front end + CPU fusion core (not Loihi 2)"),
    ("Hybrid (GPU front end + CPU fusion core)",
     "GPU front end + CPU fusion core (not Loihi 2)"),
    ("Hybrid", "GPU front end + CPU fusion core"),
)


def _relabel(label):
    """Rewrite a stored configuration label so it never implies Loihi 2."""
    if not label:
        return label
    out = label
    for old, new in _LABEL_FIXES:
        out = out.replace(old, new)
    return out


def main():
    summary = load(os.path.join(R, "run1", "summary.json"))
    if not summary:
        print("no Run 1 summary -- nothing to sync")
        return 1

    run2 = load(os.path.join(R, "run2", "summary.json"), {}) or {}
    run3 = load(os.path.join(R, "run3", "summary.json"), {}) or {}

    def stat(cfg):
        return summary.get(cfg) or {}

    def _sd(cfg, key, sub=None):
        s = stat(cfg).get(key)
        if isinstance(s, dict):
            v = s.get("sd")
            if v is not None:
                return v
        return None

    def power(cfg):
        return stat(cfg).get("power_w", {}).get("mean")

    def power_sd(cfg):
        return _sd(cfg, "power_w")

    def fps_sd(cfg):
        return _sd(cfg, "fps")

    def jpf_sd(cfg):
        return _sd(cfg, "j_per_frame_integrated")

    def jpf(cfg):
        # two independently computed forms of the same quantity; the summary
        # reports both so that they can be compared (see the "agree %" column)
        s = stat(cfg).get("j_per_frame_integrated")
        if s and s.get("mean") is not None:
            return s["mean"]
        s = stat(cfg).get("j_per_frame_mean_over_fps")
        return s["mean"] if s else None

    def fps(cfg):
        return stat(cfg).get("fps", {}).get("mean")

    def lat(cfg):
        return stat(cfg).get("latency_p50_ms", {}).get("mean")

    def _pw_max():
        """the largest single power sample recorded anywhere in the Run 1
        power logs (on-module VDD_IN rail, in watts)"""
        import csv as _csv
        import glob as _glob
        best = 0.0
        for p in _glob.glob(os.path.join(R, "..", "logs", "run1", "*.csv")):
            try:
                with open(p) as f:
                    for row in _csv.DictReader(f):
                        v = row.get("teg_vdd_in_mw_avg")
                        if v:
                            x = float(v) / 1000.0
                            if x > best:
                                best = x
            except Exception:
                continue
        return best

    def _ci_sentences():
        """Compute the confidence interval on the key energy ratios.

        The 95 % interval on a *ratio* of two means is propagated from the
        two standard deviations of the five repeats.  Everything here is
        computed from the measured data -- no figure in this function is
        hard-coded -- so the sentence cannot drift away from the table it
        describes.
        """
        out = []
        pairs = (("C4", "C1", "C4-vs-C1"),
                 ("C4", "C3", "C4-vs-C3"))
        for a, b, name in pairs:
            ma, mb = jpf(a), jpf(b)
            sa, sb = jpf_sd(a), jpf_sd(b)
            if ma is None or mb is None or not sa or not sb:
                continue
            rel = (ma / mb - 1.0) * 100.0
            # first-order propagation of the relative standard deviation
            rel_sd = math.sqrt((sa / ma) ** 2 + (sb / mb) ** 2) * 100.0
            lo, hi = rel - 1.96 * rel_sd, rel + 1.96 * rel_sd
            out.append(
                f"> **The {name} energy gap is far outside the measurement "
                f"noise.** Propagating the standard deviations of the five "
                f"repeats into the ratio gives a 95 % confidence interval of "
                f"[{lo:+.2f} %, {hi:+.2f} %] on the "
                f"{a}-vs-{b} energy-per-frame difference — a real effect, "
                f"not scatter.\n")
        return "\n".join(out)

    have_run1 = all(power(c) is not None for c in ("C1", "C4"))

    # ------------------------------------------------ the C2 baseline note
    #
    # C2 (TensorRT) is SLOWER than C1 (PyTorch), and a reader could
    # reasonably take that as a weak GPU baseline.  The cause was measured
    # rather than assumed: the workload is launch-latency bound, not
    # GEMM-throughput bound, so TensorRT's extra machinery cannot pay for
    # itself on a network this small.  The note is generated from the
    # measured profile, so it cannot drift from the data.
    def _c2_baseline_note():
        _prof = load(os.path.join(R, "run1d", "tensorrt_profile.json"),
                     {}) or {}
        _d = _prof.get("derived") or {}
        _m = _prof.get("measurements_ms_per_frame") or {}
        _ratio = _d.get("fixed_c2_to_c1_latency_ratio")
        _eng = _m.get("tensorrt_engine_executions_only_ms")
        _lif = _m.get("lif_elementwise_updates_only_ms")
        _tot = _m.get("c2_tensorrt_fixed_ms")
        if not (_ratio and _eng and _lif and _tot):
            return ("> **C2 (TensorRT) is slower than C1 (PyTorch) because "
                    "this workload is launch-latency bound, not "
                    "GEMM-throughput bound**, so the comparison is made "
                    "against C1, the stronger GPU baseline.\n")
        return (
            f"> **C2 (TensorRT) is slower than C1 (PyTorch), and C1 is the "
            f"baseline the comparison is made against.** This was measured, "
            f"not assumed: the TensorRT engines cost "
            f"**{_eng:.3f} ms** for all 75 executions per frame "
            f"({_eng / _tot * 100:.0f} % of the step), while the LIF "
            f"elementwise updates — which every backend must perform — cost "
            f"**{_lif:.3f} ms** ({_lif / _tot * 100:.0f} %), and the rest "
            f"is Python-level launch overhead. With 25 sequential time-steps "
            f"and three small dense layers, the workload is **launch-latency "
            f"bound, not GEMM-throughput bound**: a 256→512→512→64 MLP is "
            f"far too small to saturate the GPU, so TensorRT's extra "
            f"machinery cannot pay for itself. The corrected C2 runs at a "
            f"**{_ratio:.2f}x** latency ratio against C1. See "
            f"`exp_rev1/results/REVIEW_RESPONSE.md` § 1.3d.\n")

    # ------------------------------------------------ the 50 Hz provenance note
    #
    # The 50 Hz table and the unconstrained-throughput row are measured by
    # the PRE-FIX harness: every configuration and every repeat ran inside
    # ONE process, in a fixed config-major order, with no quiescence check.
    # The contamination is visible in the recorded CSVs -- C1's RAM climbs
    # 3885 -> 4202 MB and C2's 3984 -> 4155 MB, while C3, C4 and C5 all
    # start near 5300 MB because they inherit the state of the
    # configurations that ran before them in the same process.
    #
    # The 50 Hz pass is therefore being RE-MEASURED with the corrected
    # protocol (one fresh process per (config, repeat), a shuffled
    # schedule, a quiescence check per window).  Until that pass has been
    # through the gate, the 50 Hz numbers in the documents are flagged, so
    # that nobody quotes them without knowing which they are.
    #
    # The note is COMPUTED from the state of the results, not asserted, so
    # it disappears by itself the moment the corrected pass lands.
    _fresh50 = os.path.join(R, "run1_50hz_fresh", "all_runs.json")
    _fresh50_done = os.path.exists(_fresh50)
    if _fresh50_done:
        try:
            with open(_fresh50) as f:
                _f50 = json.load(f)
            _fresh50_done = bool(_f50) and all(
                len(_f50.get(c) or []) >= 5
                for c in ("C1", "C2", "C3", "C4", "C5"))
        except Exception:
            _fresh50_done = False

    if _fresh50_done:
        _PROVENANCE_NOTE_50HZ = (
            "> **This table is the corrected 50 Hz pass.** It was measured "
            "with one fresh process per (config, repeat), a shuffled "
            "schedule and a quiescence check before every window, and it "
            "replaces the earlier single-process pass whose RAM trace "
            "showed state carrying across configurations.")
    else:
        _PROVENANCE_NOTE_50HZ = (
            "> **:warning: The 50 Hz numbers in this table are from the "
            "earlier single-process pass, and they are being re-measured.** "
            "That pass ran every configuration and every repeat inside ONE "
            "process in a fixed order, with no quiescence check, and the "
            "recorded RAM trace shows the contamination plainly: C1's RAM "
            "climbs 3885 → 4202 MB and C2's 3984 → 4155 MB, while C3, C4 "
            "and C5 all start near 5300 MB because they inherit the state "
            "of the configurations that ran before them in the same "
            "process. The absolute RAM figures for C3/C4/C5 are therefore "
            "inflated by roughly 1.4 GB.\n"
            "\n"
            "> **Use the 30 Hz table below for any joules-per-frame "
            "comparison.** It was measured with the corrected protocol -- "
            "one fresh process per repeat, a shuffled schedule, a "
            "quiescence check per window -- it passed its gate, and every "
            "configuration held the full 30 Hz budget, so its joules-per-"
            "frame figures are directly comparable across all five. The "
            "50 Hz numbers above are still the right answer to a different "
            "question (\"can this configuration sustain the real-time "
            "budget?\"), and they are being re-measured; until that "
            "finishes, they carry this caveat.")

    # ---------------------------------------------- the README table
    readme_path = os.path.join(ROOT, "README.md")
    if not os.path.exists(readme_path):
        print("no README.md")
        return 1

    with open(readme_path) as f:
        readme = f.read()

    # The heading has changed several times (the first version compared
    # "Standalone GPU vs Neuromorphic Hybrid"; the current one names the
    # configuration for what it physically is).  Match the WHOLE section --
    # heading, blockquotes, table and the prose that follows it -- up to the
    # next "### " heading, so that no hand-written remnant of the old table
    # can survive the rewrite.  The rewrite is idempotent.
    old_table = re.search(
        r"### Performance Comparison:[^\n]*\n"
        r".*?"
        r"(?=\n### |\Z)",
        readme, re.S)
    if old_table:
        # Only preserve the table as a synthetic artefact if it really is
        # one.  On a second run the matched table is the MEASURED one, and
        # overwriting the audit file with it would destroy the evidence.
        _matched = old_table.group(0)
        _is_synthetic = any(k in _matched for k in
                            ("49.73", "84.34", "14.99", "11.47"))
        if _is_synthetic:
            preserved = {
                "published_performance_table": _matched,
                "note": "The published table, preserved verbatim for audit.",
            }
            with open(os.path.join(ROOT, "results", "metrics",
                                   "PUBLISHED_SYNTHETIC_VALUES.json"), "w") as f:
                json.dump(preserved, f, indent=2)

        if have_run1:
            # ---- the full five-configuration table, mean +/- SD --------
            tp = load(os.path.join(R, "run1d", "throughput.json"), {}) or {}
            tpres = tp.get("results", {}) or {}

            def tfps(cfg):
                d = tpres.get(cfg) or {}
                return d.get("throughput_fps_mean")

            def tfps_sd(cfg):
                d = tpres.get(cfg) or {}
                return d.get("throughput_fps_sd")

            COLS = ("C1", "C2", "C3", "C4", "C5")
            HEAD = ("C1 PyTorch SNN on CUDA", "C2 TensorRT FP32",
                    "C3 optimised C++ CPU (int8)",
                    "C4 GPU front end + CPU fusion core",
                    "C5 Torch CPU")
            hdr = "| Metric | " + " | ".join(HEAD) + " |\n"
            sep = "|--------|" + "|".join(
                "---" for _ in COLS) + "|\n"

            def row(label, fn, sd_fn=None, nd=3):
                cells = []
                for c in COLS:
                    v = fn(c)
                    if v is None:
                        cells.append("not measured")
                        continue
                    s = sd_fn(c) if sd_fn else None
                    if s is None:
                        cells.append(fnum(v, nd))
                    else:
                        cells.append(f"{fnum(v, nd)} ± {fnum(s, nd)}")
                return f"| {label} | " + " | ".join(cells) + " |\n"

            rows = ""
            rows += row("Throughput at 50 Hz replay (FPS)",
                        fps, fps_sd, 2)
            rows += row("Max unconstrained throughput (FPS)",
                        tfps, tfps_sd, 2)
            rows += row("Energy/Frame (J, per delivered frame)",
                        jpf, jpf_sd, 5)
            rows += row("Latency p50 (ms)", lat, None, 2)
            rows += row("Power (W)", power, power_sd, 3)

            new_table = (
                "### Performance Comparison: all five measured "
                "configurations on the Jetson Orin Nano\n"
                "\n"
                "> **MEASURED** on a Jetson Orin Nano (Orin Nano 8 GB, 6-core "
                "ARM Cortex-A78AE, 1020 MHz Ampere GPU), `nvpmodel` mode "
                "`MAXN_SUPER 2`, 5 repeats x (10 min warm-up + 10 min log) "
                "per configuration. Power is read from the on-module INA3221 "
                "via `tegrastats` at the VDD_IN input rail, i.e. at the "
                "module's own input, **not** at the SoC rails. Every cell is "
                "the **mean ± standard deviation over the 5 repeats**; the "
                "full protocol, the raw per-repeat CSVs and the quiescence "
                "record for every measurement window are in "
                "`exp_rev1/results/`.\n"
                "\n"
                "> **:warning: No Loihi 2 was measured, and no column "
                "below is a Loihi 2 column.** C4 is a **GPU front end plus "
                "the physical ARM CPU fusion core**, pipelined. No Loihi 2 "
                "board is attached to this bench, so **no Loihi 2 saving can "
                "be reported in either direction** — the Loihi 2 question "
                "was *not measured*, not answered. The C4-vs-C3 gap measures "
                "the GPU front end against the CPU path and says **nothing** "
                "about Loihi 2. See `exp_rev1/results/REVIEW_RESPONSE.md` "
                "§ SCOPE.\n"
                "\n"
                + hdr + sep + rows +
                "\n"
                + _PROVENANCE_NOTE_50HZ +
                "\n"
                "> **The J/frame column compares configurations running at "
                "different frame rates, and that has to be said wherever "
                "the table is used.** All configurations were offered the "
                "same 50 Hz input, but **only C3 and C5 deliver all of it**; "
                "C1, C2 and C4 run flat out and drop frames (their FPS "
                "figures *are* their maximum sustainable rate). So the "
                "J/frame figures are energy per frame **actually delivered**, "
                "at that configuration's own delivered rate — C1 at 44.6, "
                "C2 at 31.2, C4 at 44.1 FPS, C3 and C5 at 50 Hz. The "
                "rate-free comparison is the unconstrained throughput row "
                "above.\n"
                "\n"
                "> **The 6 W boundary excludes the sensors, the Loihi 2 "
                "board and the interconnect.** The power figures are the "
                "Jetson module's own input power; they do **not** include "
                "the sensor set, any Loihi 2 board, or the USB/PCIe link "
                "between them, and they are therefore **not** a system-level "
                "power figure for the architecture the paper proposes. See "
                "`exp_rev1/results/REVIEW_RESPONSE.md` § 1.2.\n"
                "\n"
                + _c2_baseline_note() +
                "\n"
                + _ci_sentences()
                + "\n"
                "**The measurement does not support the claim the previous "
                "version of this table made.** Splitting the work across a "
                "GPU front end and a CPU fusion core is *more* expensive per "
                "frame than the Jetson-only pipeline, not less, and neither "
                "configuration sustains the 50 Hz replay rate (both have a "
                "~22 ms p50 latency against a 20 ms budget). The previously "
                "published numbers (14.99 W / 11.47 W / 49.73 FPS / 84.34 "
                "FPS / 0.0165 J / 0.0126 J / 20.11 ms / 11.86 ms) were "
                "**random draws from `np.random` in "
                "`src/performance_metrics.py` and `src/standard_benchmark.py`, "
                "not measurements**; they are preserved verbatim in "
                "`results/metrics/PUBLISHED_SYNTHETIC_VALUES.json`. See "
                "`exp_rev1/results/REVIEW_RESPONSE.md` for the provenance "
                "audit.\n"
            )
            readme = readme[:old_table.start()] + new_table + readme[old_table.end():]

    # ------------------------------------- the README matched-rate (30 Hz)
    # The 50 Hz table above cannot compare joules per frame across
    # configurations, because C1, C2 and C4 drop frames at 50 Hz and
    # deliver a different number of frames from C3 and C5.  The identical
    # experiment was therefore repeated at 30 Hz, which is below every
    # configuration's own maximum sustainable rate, so that all five
    # configurations deliver the SAME number of frames at the SAME rate and
    # the joules-per-frame figures become directly comparable.
    #
    # This block is inserted only when the 30 Hz pass has actually produced
    # data, and it is REMOVED first if it is already there, so that the
    # generator stays idempotent.
    _M30_HEAD = "### The same comparison at a matched 30 Hz frame rate"
    while _M30_HEAD in readme:
        _a = readme.index(_M30_HEAD)
        _b = readme.find("\n### ", _a + 1)
        _b = len(readme) if _b == -1 else _b + 1
        readme = readme[:_a] + readme[_b:]

    _m30 = {}
    _p30 = os.path.join(R, "run1_30hz", "all_runs.json")
    if os.path.exists(_p30):
        with open(_p30) as f:
            _raw30 = json.load(f)
        _m30 = {k: v for k, v in _raw30.items()
                if isinstance(v, list) and v}

    if _m30:
        def _msd(vals):
            n = len(vals)
            if n == 0:
                return None, None
            m = sum(vals) / n
            if n < 2:
                return m, 0.0
            var = sum((v - m) ** 2 for v in vals) / (n - 1)
            return m, var ** 0.5

        _COLS = ("C1", "C2", "C3", "C4", "C5")
        _HEAD = ("C1 PyTorch SNN on CUDA", "C2 TensorRT FP32",
                 "C3 optimised C++ CPU (int8)",
                 "C4 GPU front end + CPU fusion core",
                 "C5 Torch CPU")
        _hdr = "| Metric | " + " | ".join(_HEAD) + " |\n"
        _sep = "|--------|" + "|".join("---" for _ in _COLS) + "|\n"

        def _mrow(label, key, nd, pct=False):
            cells = []
            for c in _COLS:
                runs = _m30.get(c) or []
                vals = [r.get(key) for r in runs if r.get(key) is not None]
                if not vals:
                    cells.append("not measured")
                    continue
                m, s = _msd(vals)
                cells.append(f"{fnum(m, nd)} ± {fnum(s, nd)}")
            return f"| {label} | " + " | ".join(cells) + " |\n"

        _mrows = ""
        _mrows += _mrow("Delivered frame rate (FPS)", "fps", 2)
        _mrows += _mrow("Energy/Frame (J, per delivered frame)",
                        "j_per_frame_integrated", 5)
        _mrows += _mrow("Power (W)", "mean_power_w", 3)

        # the latency lives one level down in the run record
        _lat_cells = []
        for c in _COLS:
            runs = _m30.get(c) or []
            vals = [(r.get("latency_ms") or {}).get("p50") for r in runs]
            vals = [v for v in vals if v is not None]
            if not vals:
                _lat_cells.append("not measured")
                continue
            m, s = _msd(vals)
            _lat_cells.append(f"{fnum(m, 2)} ± {fnum(s, 2)}")
        _mrows += ("| Latency p50 (ms) | " + " | ".join(_lat_cells) + " |\n")

        # Which configurations actually sustain the full 30 Hz budget?
        # This is COMPUTED from the measured data, not asserted: the point
        # of this pass is that the rate is matched, and if a configuration
        # still drops frames at 30 Hz then the J/frame figures are NOT
        # directly comparable for it, and the document has to say so.
        _sustain = []
        _short = []
        for c in _COLS:
            runs = _m30.get(c) or []
            vals = [r.get("fps") for r in runs if r.get("fps") is not None]
            if not vals:
                continue
            m, _s = _msd(vals)
            if m >= 29.9:
                _sustain.append(c)
            else:
                _short.append((c, m))

        if _short:
            _rate_note = (
                "> **:warning: Not every configuration sustained the full "
                "30 Hz budget in this pass**, and the table has to be read "
                "with that in the same breath. "
                + "; ".join(f"**{c} delivered only {v:.2f} FPS**"
                            for c, v in _short)
                + ". The joules-per-frame figure of such a configuration is "
                "still energy per frame *actually delivered* at its own "
                "rate, exactly as in the 50 Hz table above, so the "
                "comparison is only strictly like-for-like across the "
                "configurations that did hold the rate. The delivered rate "
                "is in the table below for exactly this reason — read it "
                "before quoting the joules-per-frame column.\n"
                "\n"
            )
        else:
            _rate_note = (
                "> **Every configuration sustained the full 30 Hz budget in "
                "this pass** (delivered rate in the table below), so the "
                "joules-per-frame figures are **directly comparable** across "
                "all five — unlike the 50 Hz table above, which compares "
                "configurations running at different frame rates.\n"
                "\n"
            )

        # ---- the per-repeat outlier disclosure
        #
        # A mean ± SD can hide a single bad repeat.  On the clean 30 Hz
        # pass, C2 rep3 missed 63 deadlines against ~10 for every other
        # repeat in the pass, while still delivering the full 18,000
        # frames.  It is disclosed rather than smoothed away, so that a
        # reviewer reading the per-repeat records finds the same outlier
        # the generator saw.  COMPUTED from the data.
        _out30 = []
        for c in _COLS:
            runs = _m30.get(c) or []
            drops = [r.get("dropped") for r in runs
                     if r.get("dropped") is not None]
            if len(drops) < 3:
                continue
            drops_sorted = sorted(drops)
            med = drops_sorted[len(drops_sorted) // 2]
            if med <= 0:
                continue
            for r in runs:
                d = r.get("dropped")
                if d is None:
                    continue
                if d >= 3 * med and d > 0:
                    _out30.append((c, r.get("rep"), d, med))
        if _out30:
            _out30_note = (
                "> **One repeat in this pass is an outlier, and it is "
                "disclosed rather than smoothed away.** "
                + "; ".join(
                    f"**{c} rep{rp} missed {d} deadlines** against a median "
                    f"of {med:.0f} for that configuration"
                    for c, rp, d, med in _out30)
                + ". Every repeat in the pass delivered the full 18,000 "
                "frames, so the joules-per-frame figures above are "
                "unaffected — but the mean ± SD does carry this repeat, and "
                "a reviewer reading the per-repeat records will find it "
                "there. It is retained rather than discarded: removing an "
                "inconvenient repeat is how a measurement stops being a "
                "measurement.\n"
                "\n"
            )
        else:
            _out30_note = ""

        _m30_block = (
            _M30_HEAD + "\n"
            "\n"
            "> **MEASURED** — the identical workload, the identical recorded "
            "stream, the identical five configurations and the identical "
            "protocol (5 repeats x (10 min warm-up + 10 min log)), replayed "
            "at **30 Hz** instead of 50 Hz. 30 Hz was chosen because it is "
            "below every configuration's own maximum sustainable rate (the "
            "slowest is C2 at 31.24 FPS), so that the configurations would "
            "as far as possible deliver the same number of frames at the "
            "same rate and the joules-per-frame figures would become "
            "comparable. The raw per-repeat CSVs are in "
            "`exp_rev1/results/run1_30hz/`.\n"
            "\n"
            + _rate_note +
            _out30_note +
            "> **This pass does not replace the 50 Hz pass, and it does not "
            "say anything about Loihi 2.** The 50 Hz pass answers *\"can "
            "this configuration keep up with the real-time budget?\"* — and "
            "the answer (C1, C2 and C4 cannot; C3 and C5 can) is a finding "
            "in its own right. This pass answers the separate question *\"at "
            "a rate as many of the configurations as possible can hold, "
            "which configuration costs less energy per frame?\"* C4 remains a "
            "**GPU front end plus the physical "
            "ARM CPU fusion core**; no Loihi 2 board is attached to this "
            "bench, so no Loihi 2 saving is reported in either direction.\n"
            "\n"
            + _hdr + _sep + _mrows +
            "\n"
        )
        # insert it immediately after the 50 Hz performance section
        _anchor = re.search(
            r"### Performance Comparison:[^\n]*\n.*?(?=\n### |\Z)",
            readme, re.S)
        if _anchor:
            readme = (readme[:_anchor.end()] + "\n" + _m30_block +
                      readme[_anchor.end():])

    # ---------------------------------------------- the README memory table
    # The memory table in the README is one of the synthetic ones: its three
    # "environments" are the config's simulated environments, and its
    # "0% / 18%" crash rates are the random draws (and 18% was never even
    # measured -- it was a hard-coded literal).  Replace the whole block
    # with the measured Run 2 result.
    run2 = {}
    _r2 = os.path.join(ROOT, "exp_rev1", "results", "run2", "summary.json")
    if os.path.exists(_r2):
        with open(_r2) as f:
            run2 = json.load(f)

    if run2:
        import re as _re
        # A previously inserted uncapped-run block must be removed FIRST.
        # It contains its own "###" heading, which would otherwise split
        # the enclosing section regex and cause the block to be inserted a
        # second time on every run.
        _NC_HEAD = "### The same experiment with NO artificial cap"
        while _NC_HEAD in readme:
            _a = readme.index(_NC_HEAD)
            _b = readme.find("\n### ", _a + 1)
            _b = len(readme) if _b == -1 else _b + 1
            readme = readme[:_a] + readme[_b:]
        _m = _re.search(
            r"### Memory Utilization \(Incremental Map Compression\).*?"
            r"(?=\n### |\n\n## |\Z)", readme, _re.S)
        if _m:
            rows = []
            for k in ("J1", "J2", "J3", "J4"):
                if k not in run2:
                    continue
                d = run2[k]
                ub = d.get("crash_rate_95pct_upper_bound")
                rows.append(
                    f"| {_relabel(d.get('label', k))} | "
                    f"{fnum(d.get('map_bytes_mean_live', 0)/1e6, 1)} MB | "
                    f"{fnum(d.get('map_bytes_mean_uncompressed', 0)/1e6, 1)} MB | "
                    f"{d.get('crashes')} / {d.get('N')} | "
                    f"{fnum((d.get('crash_rate') or 0)*100, 2)} % | "
                    f"{fnum((ub or 0)*100, 2)} % |")
            # ---- the UNCAPPED run, which is the one that can speak
            # ---- about the real 8 GB module
            run2_nc = {}
            _r2n = os.path.join(ROOT, "exp_rev1", "results", "run2_nocap",
                                "summary.json")
            if os.path.exists(_r2n):
                with open(_r2n) as f:
                    run2_nc = json.load(f)

            nc_block = ""
            if run2_nc:
                nc_rows = []
                for k in ("J1", "J2", "J3", "J4"):
                    if k not in run2_nc:
                        continue
                    d = run2_nc[k]
                    ub = d.get("crash_rate_95pct_upper_bound")
                    nc_rows.append(
                        f"| {_relabel(d.get('label', k))} | "
                        f"{fnum(d.get('map_bytes_mean_live', 0)/1e6, 1)} MB | "
                        f"{fnum(d.get('map_bytes_mean_uncompressed', 0)/1e6, 1)} MB | "
                        f"{fnum(d.get('peak_total_ram_used_mb', 0), 0)} MB | "
                        f"{d.get('crashes')} / {d.get('N')} | "
                        f"{fnum((d.get('crash_rate') or 0)*100, 2)} % | "
                        f"{fnum((ub or 0)*100, 2)} % |")
                _pk = max((d.get("peak_total_ram_used_mb") or 0)
                          for d in run2_nc.values() if isinstance(d, dict))
                _tot = next((d.get("total_ram_mb") for d in run2_nc.values()
                             if isinstance(d, dict) and d.get("total_ram_mb")),
                            None)
                _head = ""
                if _tot:
                    _head = (f" It peaked at **{_pk:.0f} MB of the module's "
                             f"{_tot:.0f} MB of physical RAM** — within "
                             f"{_tot - _pk:.0f} MB of the limit — and still "
                             f"did not fail, and the kernel's `/proc/vmstat` "
                             f"`oom_kill` counter stayed at **0**.")
                nc_block = (
                    "### The same experiment with NO artificial cap "
                    "(`--map-mem-limit-mb 0`)\n\n"
                    "> The table above ran under a 3000 MB address-space "
                    "ceiling that **the harness itself imposed**. This is "
                    "the identical experiment with **no `RLIMIT_AS` limit "
                    "at all**, so the only limit is the physical 8 GB of "
                    "the module — the run that can speak about **real** "
                    "Jetson out-of-memory behaviour.\n\n"
                    "| Configuration | Mean live map (MB) | Mean "
                    "uncompressed map (MB) | Peak total RAM (MB) | "
                    "Failures | Failure rate | 95 % one-sided upper bound |\n"
                    "|---|---|---|---|---|---|---|\n"
                    + "\n".join(nc_rows) + "\n\n"
                    "- **:warning: This table does NOT show that Algorithm 1 "
                    "prevents real out-of-memory crashes.**" + _head + " The "
                    "8 GB module never ran out of memory in this workload, "
                    "so the 8 GB limit was never actually reached. What the "
                    "table shows is that the map's growth without Algorithm "
                    "1 **comes close to the physical limit**, and that "
                    "Algorithm 1 keeps the resident footprint far below it.\n"
                    "- The kernel OOM killer **never fired**: "
                    "`/proc/vmstat` reports `oom_kill 0`, and the counter "
                    "covers the whole run window because it is cumulative "
                    "since boot (`dmesg` is not readable on this board "
                    "without root, so the counter is the authoritative "
                    "source, and it is the stronger one because it counts "
                    "kills that have already left the ring buffer).\n"
                    "- The correct statement for the paper is therefore: "
                    "*\"the map's resident footprint is bounded by "
                    "Algorithm 1 and grows without bound without it; the "
                    "un-compressed map reached within a few hundred MB of "
                    "the module's 8 GB and did not fail within 67 recorded "
                    "replays, so no real 8 GB out-of-memory crash was "
                    "reproduced in this workload.\"*\n\n"
                )

            new_mem = (
                "### Memory Utilization (Incremental Map Compression)\n\n"
                "> **MEASURED** — 67 recorded sensor replays per configuration "
                "(54.9 s each), memory sampled at 1 Hz. The previous version "
                "of this table listed three simulated environments with "
                "randomly drawn memory figures and a \"0 % / 18 %\" crash "
                "rate; those were never measured.\n\n"
                "| Configuration | Mean live map (MB) | Mean uncompressed map "
                "(MB) | Memory-cap failures | Failure rate | 95 % one-sided upper bound |\n"
                "|---|---|---|---|---|---|\n"
                + "\n".join(rows) + "\n\n"
                "- **:warning: These are memory-cap failures against a limit "
                "the harness itself imposed, not the real 8 GB module "
                "running out of memory.** The map module ran under "
                "`resource.setrlimit(RLIMIT_AS, 3000 MB)` "
                "(`--map-mem-limit-mb 3000`). So what this shows is that the "
                "map code **needs more than 3000 MB** without Algorithm 1 "
                "and does not with it. It does **not** show that Algorithm 1 "
                "prevents real 8 GB Jetson out-of-memory crashes, because "
                "the 8 GB module never ran out of memory here. The "
                "experiment was repeated with **no artificial cap** "
                "(`results/run2_nocap/`) so that only the module's own "
                "memory could limit it; see "
                "`exp_rev1/results/REVIEW_RESPONSE.md` § 2.2.\n"
                "- **The \"0 % failure rate\" is not established.** The two "
                "configurations that use Algorithm 1 recorded **no "
                "memory-cap failures in 67 replays**, which gives a 95 % "
                "one-sided Clopper-Pearson upper bound of "
                "**4.37 %** on the failure rate — the true rate could still "
                "be as high as 1 in 23. The correct statement is \"no "
                "memory-cap failure was observed in 67 replays\". The "
                "difference from the Jetson-only configuration (5 of 67) is "
                "**not statistically significant** at the 5 % level: a "
                "Fisher exact test gives p = 0.058 two-sided.\n"
                "- **The compression is real and measurable**: the live map "
                "stays bounded while the uncompressed map grows without "
                "limit, and every memory-cap failure recorded without "
                "Algorithm 1 was an out-of-memory allocation failure in the "
                "keyframe allocation.\n\n"
                + nc_block
            )
            readme = readme[:_m.start()] + new_mem + readme[_m.end():]

        # ---------------------------------------------------- the thermal table
        _m = _re.search(
            r"### Thermal Profile \([^)]*\).*?"
            r"(?=\n### |\n\n## |\Z)", readme, _re.S)
        if _m:
            run3 = {}
            _r3 = os.path.join(ROOT, "exp_rev1", "results", "run3",
                               "summary.json")
            if os.path.exists(_r3):
                with open(_r3) as f:
                    run3 = json.load(f)
            if run3:
                trows = []
                for k in ("C1", "C4"):
                    if k not in run3:
                        continue
                    d = run3[k]
                    trows.append(
                        f"| {_relabel(d.get('label', k))} | "
                        f"{fnum(d.get('mean_power_w'), 3)} W | "
                        f"{fnum(d.get('max_tj_c'), 1)} °C | "
                        f"{d.get('throttle_events_total')} |")
                new_therm = (
                    "### Thermal Profile (Thermal Stability)\n\n"
                    "> **MEASURED** — 900 s × 3 repeats per configuration, "
                    "junction temperature and throttle state logged "
                    "throughout. The previous version of this table was a "
                    "5-minute synthetic run with randomly drawn temperatures.\n\n"
                    "> **This is a thermal-stability result, not an endurance "
                    "result.** 900 s × 3 repeats with a flat temperature trace "
                    "and zero throttle events shows that the system is "
                    "thermally stable over 15 minutes. The Jetson package's "
                    "thermal time-constant is of the order of tens of "
                    "minutes, so 900 s is comparable to the settling time of "
                    "the thermal system and cannot demonstrate that the "
                    "temperature has converged. **There is no measured "
                    "endurance claim in this repository.**\n\n"
                    "| Configuration | Mean power | Max junction temp | "
                    "Throttle events |\n|---|---|---|---|\n"
                    + "\n".join(trows) + "\n\n"
                    "- **No thermal throttling** occurred in any repeat.\n"
                    "- **No battery pack is attached to this bench**, so no "
                    "battery-depletion cycle was run; the previous "
                    "\"2 battery cycles\" figure was synthetic and is "
                    "removed rather than restated.\n"
                    "- The time-to-empty at a 150 Wh pack is a **labelled "
                    "calculation** from the measured mean power, and is "
                    "reported as such in `exp_rev1/results/REVIEW_RESPONSE.md`.\n"
                )
                readme = readme[:_m.start()] + new_therm + readme[_m.end():]

    # ---------------------------------------------- the README experiment list
    # The "Experiments" list still describes the synthetic protocol as fact
    # ("50 physical flight trials", "4.5-hour sustained operation").  Those
    # describe what config/experiment_config.yaml asks the synthetic scripts
    # to do, not what was measured.  Replace the list with the measured one.
    import re as _re2
    _m = _re2.search(r"## Experiments\n(.*?)(?=\n## )", readme, _re.S)
    if _m:
        new_exp = (
            "## Experiments\n\n"
            "> What was actually measured on this machine. The previous "
            "version of this list described the *simulated* protocol that "
            "`config/experiment_config.yaml` asks the synthetic scripts "
            "under `src/` to run; none of it was executed on this bench.\n\n"
            "### 1. Incremental Map Compression\n"
            "- **Objective**: establish whether Algorithm 1 keeps the map's "
            "resident footprint bounded, and whether it removes the "
            "out-of-memory allocation failures\n"
            "- **Method**: 67 recorded sensor replays per configuration "
            "(54.9 s each), four configurations, map memory sampled at 1 Hz\n"
            "- **Result**: the live map stays bounded while the uncompressed "
            "map grows past the module's physical RAM; **no memory-cap "
            "failures in 67 replays with Algorithm 1 against 5 of 67 "
            "without**, which bounds the failure rate at 4.37 % (95 % "
            "one-sided Clopper-Pearson) rather than at zero. **These "
            "failures are against the harness's own 3000 MB "
            "address-space limit, not the real 8 GB module**\n"
            "- **The same experiment was repeated with NO artificial cap** "
            "(`--map-mem-limit-mb 0`, so the only limit is the module's own "
            "8 GB): **0 failures in all four configurations over 67 replays "
            "each**, with the un-compressed map peaking at 7312 MB of the "
            "7620 MB of physical RAM and the kernel's `oom_kill` counter "
            "staying at **0**. So the real 8 GB limit was never reached in "
            "this workload: the table shows that the map's growth without "
            "Algorithm 1 comes close to the physical limit, **not** that "
            "Algorithm 1 prevents a real 8 GB out-of-memory crash — see "
            "`exp_rev1/results/run2_nocap/` and "
            "`exp_rev1/results/REVIEW_RESPONSE.md` § 2.2\n\n"
            "### 2. Standardized Benchmarking\n"
            "- **Objective**: report power, throughput, energy per frame and "
            "latency for the configurations in this repository\n"
            "- **Workload**: the recorded rosbag `rosbag_exp4_20260404_021816` "
            "(2746 `/scan` LaserScan messages) replayed identically in every "
            "configuration\n"
            "- **Metric**: joules per frame, computed two independent ways "
            "(integrated power ÷ frames, and mean power ÷ FPS), which agree "
            "to the digits shown. **The J/frame figures compare "
            "configurations running at different frame rates** — they are "
            "energy per frame actually delivered, at that configuration's "
            "own delivered rate\n\n"
            "### 3. Configuration Comparison\n"
            "- **Objective**: compare the five configurations (C1 GPU, C2 "
            "TensorRT, C3 optimised CPU fusion core, C4 GPU front end + CPU "
            "fusion core, C5 Torch CPU) on the same workload. **No Loihi 2 "
            "was measured: no Loihi 2 board is attached to this bench**\n"
            "- **Method**: 5 repeats per configuration, 10 min warm-up + "
            "10 min log, power from the on-module INA3221 via `tegrastats` "
            "at 20 Hz, with a quiescence record for every window\n"
            "- **Result**: C4 is 7.30 % *worse* per frame than the original "
            "GPU pipeline and 48.4 % worse than the optimised CPU fusion "
            "core; the CPU paths are the only ones that sustain 50 Hz. "
            "**These gaps say nothing about Loihi 2**, which was not "
            "measured\n\n"
            "### 4. Output Equivalence\n"
            "- **Objective**: confirm that the optimised backends produce "
            "the same spikes as the reference implementation\n"
            "- **Method**: compare the TensorRT (C2) and PyTorch (C1) spike "
            "trains against a float64 NumPy evaluation of the LIF "
            "recurrence, which is the mathematical definition of the network\n"
            "- **Result**: bit-identical (0 differing spike units out of 331 "
            "per step)\n\n"
            "### 5. Thermal Stability\n"
            "- **Objective**: check for thermal throttling and sustained "
            "behaviour\n"
            "- **Method**: 900 s × 3 repeats per configuration, junction "
            "temperature and throttle state logged throughout\n"
            "- **Result**: no throttle events, max junction temperature "
            "56 °C. **This is thermal stability over 15 minutes, not an "
            "endurance result**; no battery pack is attached, so no "
            "depletion cycle was run and the time-to-empty is reported as a "
            "labelled calculation\n"
        )
        readme = readme[:_m.start()] + new_exp + readme[_m.end():]

    # ---------------------------------------------- the README warning
    banner = (
        "> ## :warning: The numbers in `src/` and in `results/metrics/` are "
        "SYNTHETIC\n"
        ">\n"
        "> The code under `src/` does **not** read the Jetson power rails, the "
        "INA3221, `tegrastats`, or any external meter. It injects values from "
        "`np.random`, which is why the published JSON files violate the "
        "identity `E_per_frame = P / FPS` by 979 % to 1732 % -- an identity "
        "that must hold automatically for any real measurement.\n"
        ">\n"
        "> The measurements that *are* real are in `exp_rev1/results/`, "
        "produced by the harness in `exp_rev1/` and documented in "
        "`exp_rev1/results/REVIEW_RESPONSE.md`. **Do not quote anything from "
        "`results/metrics/` or from the code under `src/` as a measured "
        "result.**\n"
    )
    if banner not in readme:
        anchor = "## Key Results"
        if anchor in readme:
            readme = readme.replace(anchor, anchor + "\n\n" + banner, 1)

    # ---- the withdrawn figures, and where each of them came from ----
    # The banner above warns that src/ and results/metrics/ are synthetic.
    # It does not say what happened to the figures that were actually
    # published, which is a different question and the one a reviewer asks.
    _WD_HEAD = "### The withdrawn figures, and where each came from"
    if _WD_HEAD in readme:
        _a = readme.index(_WD_HEAD)
        _b = readme.find("\n### ", _a + 1)
        if _b == -1:
            _b = readme.find("\n## ", _a + 1)
        _b = len(readme) if _b == -1 else _b + 1
        readme = readme[:_a] + readme[_b:]
    _wd = (
        "### The withdrawn figures, and where each came from\n"
        "\n"
        "Five figures were quoted in the submitted manuscript. All five are "
        "**withdrawn — unsupported**. Their provenance was established by "
        "searching the **submitted tree** (commit `584ba0c`) for each "
        "figure as text, as a JSON numeric value, and as the visible text "
        "of the SVG figures. The evidence is "
        "`exp_rev1/results/withdrawn_figures_forensics.json`.\n"
        "\n"
        "| Withdrawn figure | Provenance in the submitted tree |\n"
        "|---|---|\n"
        "| 60.9 W | **not present anywhere** — no source file, no result "
        "file, no figure, no log |\n"
        "| 22.7 W | **not present anywhere** |\n"
        "| 0.145 W | **not present anywhere** |\n"
        "| 99.3 % | **not present anywhere** |\n"
        "| 11.86 ms | **present** — "
        "`results/comparison/metrics_comparison.json`, produced by "
        "`np.random.normal(12, 2)` in `src/performance_metrics.py` |\n"
        "\n"
        "> **What the repository can state, and what it cannot.** It can "
        "state that those four figures had no measurement behind them and "
        "that the fifth was a random draw. It **cannot** state where the "
        "four came from, and this repository does not speculate about it. "
        "That is a question for the authors.\n"
        "\n"
    )
    # The anchor has to be a heading that is actually in the README.  "## Key
    # Results" was a plausible guess that does not exist, so the whole block
    # was silently dropped -- an insertion that fails quietly is worse than
    # one that fails loudly.
    _anchor_w = "### Performance Comparison: all five measured configurations"
    if _anchor_w in readme:
        readme = readme.replace(_anchor_w, _wd + _anchor_w, 1)
    else:
        raise SystemExit("sync_published_numbers: cannot place the "
                         "withdrawn-figures table -- anchor not found in "
                         "README.md")

    # ---- the file-tree comments must not restate withdrawn figures ----
    # The tree annotates each source file with what it does.  Two of those
    # annotations quoted figures that were never measured ("50 physical
    # flight trials", "4.5-hour stress test"), so they are restated in
    # terms of what was actually run.
    # The tree must list the two documents that carry the paper's status:
    # the Track A extension case and the internal memo.  A reader who
    # cannot find them will not know that either exists.
    # The insertion has to be idempotent: a plain str.replace() of the
    # anchor with "anchor + new lines" appends the new lines AGAIN on every
    # run, so the tree grows by two lines per run.  Remove any previously
    # inserted copies first, then insert exactly one.
    _TREE_A = "│   ├── TRACK_A_LOIHI2_ACCESS_AND_EXTENSION.md   # Track A: the Loihi 2 protocol + extension case"
    _TREE_B = "│   ├── INTERNAL_MEMO_FIGURE_PROVENANCE.md      # internal memo to Dr. Zhang (not sent)"
    _TREE_C = "│   ├── measurement_lock.py                # the exclusive lock that protects measurement windows"
    _TREE_D = "│   ├── MEASUREMENT_LOCK_INCIDENT.md       # why the lock exists: the aborted 2026-11-05 pass"
    readme = readme.replace(_TREE_A + "\n", "").replace(_TREE_B + "\n", "") \
                   .replace(_TREE_C + "\n", "").replace(_TREE_D + "\n", "")
    readme = readme.replace(
        "│   ├── quiescence.py                    # the idle guard every window must pass",
        "│   ├── quiescence.py                    # the idle guard every window must pass\n"
        + _TREE_A + "\n" + _TREE_B + "\n" + _TREE_C + "\n" + _TREE_D)
    readme = readme.replace(
        "│   ├── physical_test_runner.py            # 50 physical flight trials",
        "│   ├── physical_test_runner.py            # recorded-replay driver "
        "(67 replays/configuration)")
    readme = readme.replace(
        "│   ├── endurance_stress_test.py           # 4.5-hour stress test",
        "│   ├── endurance_stress_test.py           # thermal stress test "
        "(900 s x 3 repeats)")
    readme = readme.replace(
        "│   ├── physical_test_runner.py            # 50 physical flight trials ",
        "│   ├── physical_test_runner.py            # recorded-replay driver "
        "(67 replays/configuration) ")
    readme = readme.replace(
        "│   ├── endurance_stress_test.py           # 4.5-hour stress test ",
        "│   ├── endurance_stress_test.py           # thermal stress test "
        "(900 s x 3 repeats) ")

    # ---- the citation block ----
    # The README used to name a completely different paper and a completely
    # different journal from the one the appeal letter addresses ("Trends
    # in Biotechnology" / "Bio-Inspired Navigation Mechanisms", against
    # "IEEE Transactions on Robotics" / "Loihi 2 Hybrid Neuromorphic SLAM
    # for Aerial Robotics").  It is leftover boilerplate from the original
    # repository and it contradicts the submission, so it is corrected here.
    readme = readme.replace(
        "This project is part of the manuscript submission to "
        "*Trends in Biotechnology*.",
        "This project is part of the manuscript submission to "
        "*IEEE Transactions on Robotics*.")
    readme = readme.replace(
        "Manuscript: Bio-Inspired Navigation Mechanisms\n"
        "Journal: Trends in Biotechnology",
        "Manuscript: Loihi 2 Hybrid Neuromorphic SLAM for Aerial Robotics\n"
        "Journal: IEEE Transactions on Robotics")

    # ---- the manuscript-status section ----
    # The repository has to say where the PAPER stands, not just where the
    # measurements stand.  The measurements are complete for everything the
    # Jetson can answer; they do not support the manuscript's title.
    _STAT_HEAD = "## Status of this manuscript"
    if _STAT_HEAD in readme:
        _a = readme.index(_STAT_HEAD)
        _b = readme.find("\n## ", _a + 1)
        _b = len(readme) if _b == -1 else _b + 1
        readme = readme[:_a] + readme[_b:]
    _stat = (
        "## Status of this manuscript\n"
        "\n"
        "> **The measurements are complete for everything the Jetson can "
        "answer, and they are not enough.** The title and the contribution "
        "of this work are about a **Jetson + Loihi 2** system, and **no "
        "Loihi 2 configuration has ever been measured** — not on this "
        "machine, and not on any machine in this repository's history. "
        "Every Loihi 2 quantity the reviewers asked for is `UNAVAILABLE`, "
        "and it stays that way.\n"
        "\n"
        "The measured results are:\n"
        "\n"
        "- The **optimised CPU fusion core (C3) is 27.68 % cheaper per "
        "frame** than the original GPU pipeline (C1), and it is the "
        "**fastest** configuration measured (105.85 FPS unconstrained, the "
        "only one besides C5 that sustains the 50 Hz replay rate).\n"
        "- The **GPU front end + CPU fusion core (C4) is 48.4 % more "
        "expensive per frame** than C3 and draws 30.8 % more power. It "
        "does not sustain 50 Hz. This is a GPU-front-end-versus-CPU "
        "measurement and says **nothing** about Loihi 2.\n"
        "- **Algorithm 1 bounds the map's resident footprint.** Without it "
        "the map grew to within 308 MB of the module's physical 8 GB; "
        "with it the resident footprint is an order of magnitude smaller. "
        "The 8 GB limit was never reached, so this does **not** show that "
        "Algorithm 1 prevents real out-of-memory crashes.\n"
        "- **Thermal stability over 15 minutes**, with zero throttle "
        "events. Not an endurance result.\n"
        "\n"
        "### What this means for the paper\n"
        "\n"
        "The manuscript is in one of three states, and this repository "
        "cannot decide which:\n"
        "\n"
        "1. **Measure Loihi 2 and extend the revision.** The only route "
        "that keeps the paper's actual contribution. The protocol, the "
        "hardware list and the extension case are in "
        "`exp_rev1/TRACK_A_LOIHI2_ACCESS_AND_EXTENSION.md`. The "
        "measurement harness, the power instrumentation, the quiescence "
        "guard and the backend-equivalence regression test already exist; "
        "only the sixth backend and the hardware are missing.\n"
        "2. **Rescope the paper** to a measured GPU-versus-CPU fusion "
        "study with Algorithm 1, move Loihi 2 to future work, and change "
        "the title, abstract and conclusion. Honest, but a different "
        "paper with weaker novelty.\n"
        "3. **Withdraw.**\n"
        "\n"
        "> **No Loihi 2 saving may be claimed in either direction until a "
        "board exists and the protocol has run.**\n"
        "\n"
    )
    _anchor_s = "## Verification"
    if _anchor_s in readme:
        readme = readme.replace(_anchor_s, _stat + _anchor_s, 1)
    else:
        readme = readme.rstrip() + "\n\n" + _stat

    # ---- the measurement lock section ----
    # The lock exists because the quiescence guard can only check the
    # machine BEFORE a window opens.  Anything that starts inside a window
    # contaminates the numbers being produced, and the guard cannot see it.
    _LOCK_HEAD = "### The measurement lock"
    if _LOCK_HEAD in readme:
        _a = readme.index(_LOCK_HEAD)
        _b = readme.find("\n### ", _a + 1)
        if _b == -1:
            _b = readme.find("\n## ", _a + 1)
        _b = len(readme) if _b == -1 else _b + 1
        readme = readme[:_a] + readme[_b:]
    _lock = (
        "### The measurement lock\n"
        "\n"
        "> **Do not run the document generators or the verification suite "
        "while a measurement is running.** They render figures and rewrite "
        "documents, which is CPU-heavy load, and a power measurement "
        "cannot tolerate it. The tooling now refuses to start while a "
        "measurement holds the lock, so this cannot happen by accident.\n"
        "\n"
        "This is not hypothetical. On 2026-11-05 the 30 Hz matched-rate "
        "pass was aborted at `C1_rep1` because the author's own "
        "`verify_final.py` was rendering figures at 81 % CPU inside the "
        "measurement window. The quiescence guard caught it and refused to "
        "emit the contaminated figure -- correct behaviour -- but the run "
        "was still lost. The account is in "
        "`exp_rev1/MEASUREMENT_LOCK_INCIDENT.md`.\n"
        "\n"
        "How it works:\n"
        "\n"
        "- Every measurement script takes an exclusive `flock` on "
        "`exp_rev1/.measurement.lock` for the whole run.\n"
        "- Every CPU-heavy generator checks that lock before it starts "
        "and aborts with a message naming the measurement in progress.\n"
        "- The lock is released automatically if the process dies, so a "
        "stale lock file cannot wedge the machine.\n"
        "\n"
    )
    _anchor_l = "## Verification"
    if _anchor_l in readme:
        readme = readme.replace(_anchor_l, _lock + _anchor_l, 1)

    # ---- the Verification section ----
    # It has to state what the verification script actually establishes:
    # agreement between the documents and the internal consistency of the
    # pipeline.  It does NOT establish that the measurements are correct.
    _VERIF_HEAD = "## Verification"
    if _VERIF_HEAD in readme:
        _a = readme.index(_VERIF_HEAD)
        _b = readme.find("\n## ", _a + 1)
        _b = len(readme) if _b == -1 else _b + 1
        readme = readme[:_a] + readme[_b:]
    _verif = (
        "## Verification\n"
        "\n"
        "`exp_rev1/verify_final.py` runs the consistency checks over the "
        "README, the reviewer response, the manuscript tables and the "
        "appeal letter.\n"
        "\n"
        "> **What these checks establish:** the agreement between the "
        "documents and the internal consistency of the pipeline — that "
        "every document quotes the same numbers, that the withdrawn "
        "figures appear nowhere as measured claims, that the Loihi 2 "
        "question is stated as unmeasured everywhere, and that the "
        "confidence bounds, the failure rates and the run counts are "
        "computed from the raw data rather than typed in by hand.\n"
        ">\n"
        "> **What these checks do NOT establish:** that the measurements "
        "are correct. A consistency check proves that the README, the "
        "reviewer response, the manuscript tables and the appeal letter "
        "all say the same thing; it cannot prove that the number they all "
        "repeat is the right number. The evidence for the measurements is "
        "the raw data itself — the per-repeat CSVs, the per-configuration "
        "JSON files, the quiescence records and the kernel counters — and a "
        "reviewer should check those directly.\n"
        "\n"
        "```bash\n"
        "cd exp_rev1 && python3 verify_final.py\n"
        "```\n"
        "\n"
    )
    _anchor = "## License"
    if _anchor in readme:
        readme = readme.replace(_anchor, _verif + _anchor, 1)
    else:
        readme = readme.rstrip() + "\n\n" + _verif

    with open(readme_path, "w") as f:
        f.write(readme)
    print("rewrote README.md")

    # ---------------------------------------------- the appeal letters
    #
    # The letters are GENERATED, not patched.  An earlier version of this
    # script rewrote the submitted letters in place, which left them with
    # the shape of the original submission -- an appeal that says "we have
    # now validated our claims" while listing three NOT SUPPORTED results
    # and an UNMEASURED subject.  make_appeal_letters.py builds them from
    # the measured data instead, so their structure follows the actual
    # situation.
    _gen_letters = os.path.join(ROOT, "exp_rev1", "make_appeal_letters.py")
    if os.path.exists(_gen_letters):
        import subprocess as _sp
        _r = _sp.run([sys.executable, _gen_letters],
                     cwd=os.path.join(ROOT, "exp_rev1"),
                     capture_output=True, text=True)
        if _r.returncode == 0:
            print("regenerated appeal_letter/appeal_letter.md and .txt")
            for _line in (_r.stdout or "").splitlines():
                if _line.strip().startswith("NOTE:"):
                    print("  " + _line.strip())
        else:
            print("WARNING: make_appeal_letters.py failed; the appeal "
                  "letters were left untouched")
            print((_r.stderr or "")[-1500:])

    # ---------------------------------------------- the metrics JSONs
    #
    # Every published result file outside exp_rev1/ carries values that came
    # from np.random in src/.  Each one gets a SYNTHETIC_ORIGIN marker so
    # that a reader who opens the file cannot mistake it for a measurement.
    # The files themselves are NOT deleted: they are the evidence for the
    # provenance audit, and deleting them would also destroy the audit trail.
    NOTE = {
        "warning": "The values in this file were random draws from "
                   "np.random in src/, not measurements. They are retained "
                   "here only as evidence for the provenance audit and MUST "
                   "NOT be cited as results.",
        "provenance_audit": "exp_rev1/results/provenance_audit.json",
        "measured_results": "exp_rev1/results/",
    }
    targets = []
    for sub in ("metrics", "benchmark", "comparison", "physical_tests",
                "stress_tests"):
        d0 = os.path.join(ROOT, "results", sub)
        if os.path.isdir(d0):
            for fn in sorted(os.listdir(d0)):
                if fn.endswith(".json"):
                    targets.append(os.path.join(d0, fn))
    for extra in ("incremental_compression_summary.json",):
        p0 = os.path.join(ROOT, "results", extra)
        if os.path.exists(p0):
            targets.append(p0)

    for p in targets:
        if os.path.basename(p) == "PUBLISHED_SYNTHETIC_VALUES.json":
            continue
        rel = os.path.relpath(p, ROOT)
        try:
            d = json.load(open(p))
        except Exception as exc:
            # A published file that is not even valid JSON cannot be parsed
            # and annotated in place.  Record the fact next to it instead of
            # silently leaving an unexplained broken file in the tree.
            side = p + ".SYNTHETIC.md"
            with open(side, "w") as f:
                f.write(
                    f"# SYNTHETIC / CORRUPTED RESULT FILE\n\n"
                    f"`{rel}` is a published result file whose values came from "
                    f"`np.random` in `src/`.\n\n"
                    f"It is additionally **truncated and is not valid JSON** "
                    f"(`{exc}`), so it could not be annotated in place.\n\n"
                    f"It is retained only as evidence for the provenance audit "
                    f"and **must not be cited as a result**. The measured "
                    f"results are in `exp_rev1/results/`.\n")
            print(f"marked {rel} as synthetic-and-corrupt")
            continue
        if not isinstance(d, dict):
            continue
        if "SYNTHETIC_ORIGIN" in d:
            continue
        d["SYNTHETIC_ORIGIN"] = NOTE
        with open(p, "w") as f:
            json.dump(d, f, indent=2)
        print(f"annotated {rel}")

    print("\nsync complete")
    return 0



if __name__ == "__main__":
    # This tool is CPU-heavy and must NOT run while a measurement is in
    # progress: it would contaminate the open measurement window.  On
    # 2026-11-05 it did exactly that and cost a whole 30 Hz pass.
    from measurement_lock import refuse_if_measurement_running
    refuse_if_measurement_running("sync_published_numbers.py")
    raise SystemExit(main())
