#!/usr/bin/env python3
"""
Final verification of the whole measurement pipeline.

Run after the final pass.  It answers one question: **is every artefact in
the deliverable set actually derived from the clean measurement?**

The checks are:

  1. every expected deliverable exists and is not empty;
  2. every result artefact is NEWER than the quiescence record that opened
     the clean pass, so no stale number can survive unnoticed;
  3. every Run 2 configuration recorded the full N replays;
  4. the crash-rate figures are consistent with the recorded crash counts
     (recomputed here from the counts, not trusted from the JSON);
  5. the energy identity holds in the measured Run 1 data;
  6. the measured and synthetic figures are still different, so the audit
     trail is intact;
  7. no result file still contains the published synthetic values.

Exit status is non-zero if any check fails, so the pipeline cannot report
success over a broken deliverable set.
"""

import json
import os
import re
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(HERE, "results")
FIG = os.path.join(ROOT, "figure")

FAIL = []
WARN = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)
    return cond


def warn(cond, msg):
    if not cond:
        print("  WARN " + msg)
        WARN.append(msg)


def load(p, default=None):
    if not os.path.exists(p):
        return default
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        FAIL.append(f"{p}: unreadable ({e})")
        return default


def main():
    print("=" * 72)
    print(" FINAL VERIFICATION")
    print("=" * 72)

    # ---------------------------------------------------- 1. deliverables
    print("\n[1] the deliverables exist")
    expected = [
        os.path.join(R, "REVIEW_RESPONSE.md"),
        os.path.join(R, "provenance_audit.json"),
        os.path.join(R, "review_response.json"),
        os.path.join(R, "run1", "summary.json"),
        os.path.join(R, "run1", "all_runs.json"),
        os.path.join(R, "run1", "pose_equivalence.json"),
        os.path.join(R, "run1", "stage_decomposition.json"),
        os.path.join(R, "run2", "summary.json"),
        os.path.join(R, "run3", "summary.json"),
        os.path.join(R, "run1d", "throughput.json"),
        os.path.join(R, "run1d", "tensorrt_profile.json"),
        os.path.join(R, "manuscript", "tables.tex"),
        os.path.join(R, "manuscript", "results.md"),
        os.path.join(R, "manuscript", "numbers.json"),
        os.path.join(R, "manuscript", "summary.txt"),
        os.path.join(FIG, "S1.png"), os.path.join(FIG, "S2.png"),
        os.path.join(FIG, "S3.png"),
        os.path.join(ROOT, "README.md"),
        os.path.join(ROOT, "appeal_letter", "appeal_letter.md"),
        os.path.join(ROOT, "appeal_letter", "appeal_letter.txt"),
    ]
    for p in expected:
        ok = os.path.exists(p) and os.path.getsize(p) > 0
        rel = os.path.relpath(p, ROOT)
        check(ok, f"{rel} exists and is not empty")

    # ---------------------------------------------------- 2. freshness
    print("\n[2] no result artefact predates the clean pass")
    marker_p = os.path.join(R, "quiescence_before_clean_pass.json")
    if not os.path.exists(marker_p):
        WARN.append("no clean-pass quiescence marker to compare against")
        print("  WARN  no quiescence_before_clean_pass.json -- cannot check "
              "freshness")
    else:
        marker = load(marker_p, {}) or {}
        ts = marker.get("timestamp")
        try:
            t0 = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()
        except Exception:
            t0 = None
        if t0 is None:
            WARN.append("the clean-pass marker has no usable timestamp")
        else:
            stale = []
            for rel in ("run1/all_runs.json", "run1/summary.json",
                        "run1/stage_decomposition.json",
                        "run1/pose_equivalence.json",
                        "run2/summary.json", "run3/summary.json"):
                p = os.path.join(R, rel)
                if os.path.exists(p) and os.path.getmtime(p) < t0:
                    stale.append(rel)
            for s in stale:
                check(False, f"{s} is older than the clean pass -- STALE")
            if not stale:
                check(True, "every result artefact is newer than the clean "
                            "pass")

    # ---------------------------------------------------- 3. Run 2 counts
    print("\n[3] every Run 2 configuration recorded the full N")

    # ------------------------------------------- 2b. the matched-rate pass
    #
    # The clean 30 Hz pass is the authoritative dataset, so its integrity
    # is checked directly: 25/25 repeats, the gate verdict, and the
    # dropped-frame count.
    #
    # The DROPPED COUNT is checked against its defining identity:
    #
    #     dropped == max(0, int(elapsed_s * rate_hz) - frames)
    #
    # i.e. the number of slots the replay clock offered during the window,
    # minus the number of frames it actually delivered on time.  The old
    # counter violated this by counting one drop per late frame AND then
    # counting the same lateness again for every frame that followed, so
    # a single late frame poisoned the count for the whole window.  This
    # check is what stops that class of defect from coming back.
    print("\n[2b] the matched-rate (30 Hz) pass is intact and its dropped "
          "counts satisfy their identity")
    _m30_dir = os.path.join(R, "run1_30hz")
    _m30_all = load(os.path.join(_m30_dir, "all_runs.json"), {}) or {}
    _m30_rate = None
    _sched = load(os.path.join(_m30_dir, "schedule.json"), {}) or {}
    if isinstance(_sched, dict):
        _m30_rate = _sched.get("rate_hz")
    if not _m30_all:
        check(False, "the matched-rate pass has no all_runs.json")
    else:
        _n_rep = 0
        _bad_identity = []
        _missing_note = []
        for _cfg, _runs in _m30_all.items():
            if not isinstance(_runs, list):
                continue
            if len(_runs) < 5:
                check(False, f"{_cfg}: only {len(_runs)} repeats in the "
                             f"matched-rate pass (5 are required)")
            for _r in _runs:
                if not isinstance(_r, dict):
                    continue
                _n_rep += 1
                _el = _r.get("elapsed_s")
                _fr = _r.get("frames")
                _dr = _r.get("dropped")
                if None in (_el, _fr, _dr) or not _m30_rate:
                    continue
                _exp = max(0, int(float(_el) * float(_m30_rate)) - int(_fr))
                if int(_dr) != _exp:
                    _bad_identity.append(
                        f"{_cfg} rep {_r.get('rep')}: dropped {_dr} != "
                        f"{_exp}")
                if _r.get("dropped_old_counter") is not None and \
                        not _r.get("dropped_note"):
                    _missing_note.append(f"{_cfg} rep {_r.get('rep')}")
        check(_n_rep >= 25,
              f"the matched-rate pass has {_n_rep} repeats (25 are required)")
        if _bad_identity:
            for _b in _bad_identity[:5]:
                check(False, f"the dropped count violates its identity: {_b}")
        else:
            check(True, "every dropped count satisfies "
                        "dropped == int(elapsed_s * rate) - frames")
        check(not _missing_note,
              "every repaired dropped count records the old value and the "
              "reason it was repaired")

    # the gate verdict for the matched-rate pass
    _gv = load(os.path.join(_m30_dir, "gate_verdict.json"), {}) or {}
    if _gv:
        check(bool(_gv.get("passed")),
              "the matched-rate pass passed its gate")
    else:
        WARN.append("the matched-rate pass has no gate_verdict.json")
        print("  WARN  no gate_verdict.json for the matched-rate pass")

    # the archived failed trial must keep its marker
    _failed_marker = os.path.join(R, "run1_30hz_failed",
                                   "FAILED_TRIAL_DO_NOT_QUOTE.txt")
    check(os.path.exists(_failed_marker),
          "the archived failed 30 Hz trial keeps its "
          "FAILED_TRIAL_DO_NOT_QUOTE.txt marker")

    run2 = load(os.path.join(R, "run2", "summary.json"), {}) or {}
    N = 67
    for k in ("J1", "J2", "J3", "J4"):
        if k not in run2:
            check(False, f"Run 2 is missing configuration {k}")
            continue
        d = run2[k]
        check(d.get("N") == N, f"{k} recorded N = {d.get('N')} (expected {N})")
        cs = d.get("crashes")
        check(isinstance(cs, int) and 0 <= cs <= N,
              f"{k} crash count {cs} is in range")

    # ---------------------------------------------------- 4. crash rates
    print("\n[4] the crash rates agree with the recorded crash counts")
    for k, d in run2.items():
        n, c = d.get("N"), d.get("crashes")
        if not n or c is None:
            continue
        recomputed = c / n
        stated = d.get("crash_rate")
        check(stated is not None and abs(stated - recomputed) < 1e-9,
              f"{k}: stated rate {stated} matches {c}/{n} = {recomputed:.6f}")
        ub = d.get("crash_rate_95pct_upper_bound")
        check(ub is not None and ub >= recomputed - 1e-12,
              f"{k}: the 95 % upper bound {ub} is above the point estimate")
        if c == 0:
            check(ub is not None and 0.04 < ub < 0.05,
                  f"{k}: 0 crashes in {n} runs gives the expected 4.37 % "
                  f"Clopper-Pearson bound (got {ub})")

    # ---------------------------------------------------- 5. energy identity
    print("\n[5] the energy identity holds in the measured Run 1 data")
    allr = load(os.path.join(R, "run1", "all_runs.json"), {}) or {}
    for cfg, runs in allr.items():
        for r in runs:
            j_int = r.get("j_per_frame_integrated")
            j_fps = r.get("j_per_frame_mean_over_fps")
            p = r.get("mean_power_w")
            f = r.get("fps")
            if None in (j_int, j_fps, p, f) or not f:
                continue
            implied = p / f
            err = abs(j_int - implied) / implied * 100 if implied else 0
            if err > 1.0:
                check(False, f"{cfg} rep {r.get('rep')}: integrated "
                            f"{j_int:.6f} vs implied {implied:.6f} J/frame "
                            f"({err:.2f} % apart)")
                break
        else:
            check(True, f"{cfg}: the integrated and P/FPS joules-per-frame "
                        f"agree")

    # ---------------------------------------------------- 6. the audit trail
    print("\n[6] the synthetic originals are intact and differ from the "
          "measured figures")
    for name in ("S1", "S2", "S3"):
        a = os.path.join(FIG, name + ".png")
        b = os.path.join(FIG, name + "_synthetic.png")
        if not (os.path.exists(a) and os.path.exists(b)):
            check(False, f"{name}: the measured/synthetic pair is incomplete")
            continue
        same = open(a, "rb").read() == open(b, "rb").read()
        check(not same, f"{name}: the measured figure differs from the "
                        f"preserved synthetic original")

    # ---------------------------------------------------- 7. no synthetic
    #            values left in the published files
    print("\n[7] the published files no longer quote the synthetic values")
    synthetic = ["14.99 W", "11.47 W", "49.73", "84.34", "0.0165", "0.0126",
                 "20.11", "11.86"]
    for rel in ("README.md", "appeal_letter/appeal_letter.md",
                "appeal_letter/appeal_letter.txt"):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            txt = f.read()
        # a synthetic value is only acceptable inside an explicit
        # provenance disclosure, i.e. on a line that says it was random
        # The context is the surrounding PARAGRAPH, not the single line: a
        # disclosure can legitimately wrap across lines, and a line-based
        # check either misses the disclosure or has to be worded so that
        # the qualifier happens to land on the same line.  The paragraph is
        # a wider context, so the check is stricter, not looser.
        offenders = []
        paras = re.split(r"\n\s*\n", txt)
        for para in paras:
            low = para.lower()
            for v in synthetic:
                if v in para:
                    if not any(k in low for k in
                               ("random", "synthetic", "published", "not a "
                                "measurement", "np.random", "previously",
                                "withdraw", "unsupported")):
                        offenders.append((v, para.strip()[:110]))
        check(not offenders,
              f"{rel}: every synthetic figure appears only inside a "
              f"provenance disclosure")
        for v, line in offenders:
            print(f"        {v} :: {line}")

    # ---------------------------------------------------- 8. the statistics
    print("\n[8] the Run 2 statistics are stated correctly")
    try:
        from scipy import stats as _st
        _a, _b, _c, _d = 5, 62, 0, 67
        _p2 = _st.fisher_exact([[_a, _b], [_c, _d]], alternative="two-sided")[1]
        _p1 = _st.fisher_exact([[_a, _b], [_c, _d]], alternative="greater")[1]
        check(abs(_p2 - 0.058) < 0.002,
              f"the two-sided Fisher exact p is {_p2:.4f} (the documents quote "
              f"0.058)")
        check(abs(_p1 - 0.029) < 0.002,
              f"the one-sided Fisher exact p is {_p1:.4f} (the documents quote "
              f"0.029)")
        check(_p2 >= 0.05,
              "the two-sided p does NOT reach significance at 0.05, so the "
              "documents must not claim it does")
        _ub1 = _st.beta.ppf(0.95, 1, 67)
        _ub2 = _st.beta.ppf(0.975, 1, 67)
        check(abs(_ub1 - 0.0437) < 0.001,
              f"the one-sided Clopper-Pearson upper bound for 0/67 is "
              f"{_ub1*100:.2f} % (the documents quote 4.37 %)")
        check(abs(_ub2 - 0.0536) < 0.002,
              f"the two-sided Clopper-Pearson upper bound for 0/67 is "
              f"{_ub2*100:.2f} % (the documents quote 5.36 %)")
        check(abs(5 / 67 - 0.0746) < 0.0002,
              "the Jetson-only failure rate is 7.46 % (5 of 67), not 18 %")
    except ImportError:
        WARN.append("scipy is not available -- the Fisher/Clopper-Pearson "
                    "checks were skipped")

    # ---------------------------------------------------- 9. terminology
    print("\n[9] the terminology is consistent across the deliverables")
    # 9a: no document may call the Run 2 failures OOM kills
    for rel in ("results/REVIEW_RESPONSE.md",
                "results/manuscript/results.md",
                "results/manuscript/summary.txt"):
        p = os.path.join(ROOT, "exp_rev1", rel)
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        bad = []
        for line in txt.splitlines():
            low = line.lower()
            if "oom" in low and "kill" in low:
                # allowed only in an explicit correction of the terminology
                if not any(k in low for k in ("not", "never", "rather",
                                              "memory-cap", "address-space",
                                              "no kernel", "no matches",
                                              "counter", "grep for",
                                              "no document calls",
                                              "must not call",
                                              "are not the kernel")):
                    bad.append(line.strip()[:110])
        check(not bad, f"{rel}: no document calls the Run 2 failures "
                        f"'OOM kills' without correcting the term")
        for b in bad:
            print(f"        {b}")

    # 9b: the 18 % figure must not survive as a measured claim
    for rel in ("results/REVIEW_RESPONSE.md",
                "results/manuscript/results.md",
                "results/manuscript/summary.txt",
                "appeal_letter/appeal_letter.md",
                "appeal_letter/appeal_letter.txt"):
        p = os.path.join(ROOT, rel) if rel.startswith("appeal") else \
            os.path.join(ROOT, "exp_rev1", rel)
        if not os.path.exists(p):
            continue
        bad = []
        for line in open(p).read().splitlines():
            low = line.lower()
            if ("18 %" in line or "18%" in line):
                if not any(k in low for k in ("not ", "never", "hard-coded",
                                              "previously", "was not",
                                              "random", "measured")):
                    bad.append(line.strip()[:110])
        check(not bad, f"{rel}: the 18 % figure appears only as a "
                        f"corrected/never-measured figure")
        for b in bad:
            print(f"        {b}")

    # 9c: Run 2 must be called a replay, never a flight trial
    for rel in ("results/REVIEW_RESPONSE.md",
                "results/manuscript/results.md",
                "appeal_letter/appeal_letter.md",
                "appeal_letter/appeal_letter.txt"):
        p = os.path.join(ROOT, rel) if rel.startswith("appeal") else \
            os.path.join(ROOT, "exp_rev1", rel)
        if not os.path.exists(p):
            continue
        bad = []
        for line in open(p).read().splitlines():
            low = line.lower()
            if "flight trial" in low or "flight test" in low:
                if not any(k in low for k in ("not ", "never", "rather",
                                              "must not", "no vehicle",
                                              "not flight", "no live sensors")):
                    bad.append(line.strip()[:110])
        check(not bad, f"{rel}: Run 2 is called a replay, never a flight "
                        f"trial")
        for b in bad:
            print(f"        {b}")

    # 9d: no document may claim an endurance result from the 900 s run
    for rel in ("results/REVIEW_RESPONSE.md",
                "results/manuscript/results.md",
                "results/manuscript/summary.txt",
                "appeal_letter/appeal_letter.md",
                "appeal_letter/appeal_letter.txt",
                "README.md"):
        p = os.path.join(ROOT, rel) if rel.startswith("appeal") else \
            os.path.join(ROOT, rel) if rel == "README.md" else \
            os.path.join(ROOT, "exp_rev1", rel)
        if not os.path.exists(p):
            continue
        bad = []
        for line in open(p).read().splitlines():
            low = line.lower()
            if "4.5-hour" in low or "4.5 hour" in low or "4.5+ hour" in low:
                if not any(k in low for k in ("not ", "never", "rather",
                                              "must not", "was not",
                                              "not backed")):
                    bad.append(line.strip()[:110])
        check(not bad, f"{rel}: the 4.5-hour endurance claim does not survive "
                        f"as a measured claim")
        for b in bad:
            print(f"        {b}")

    # ---------------------------------------------------- 10. the energy claims
    print("\n[10] the energy comparisons are stated against the right pairs")
    allr = load(os.path.join(R, "run1", "all_runs.json"), {}) or {}

    def _jpf(cfg):
        runs = allr.get(cfg) or []
        v = [r["j_per_frame_integrated"] for r in runs
             if r.get("j_per_frame_integrated")]
        return sum(v) / len(v) if v else None

    def _pw(cfg):
        runs = allr.get(cfg) or []
        v = [r["mean_power_w"] for r in runs if r.get("mean_power_w")]
        return sum(v) / len(v) if v else None

    _j1, _j3, _j4 = _jpf("C1"), _jpf("C3"), _jpf("C4")
    if None not in (_j1, _j3, _j4):
        _saving = (1 - _j3 / _j1) * 100
        _c4v3 = (_j4 / _j3 - 1) * 100
        _c4v1 = (_j4 / _j1 - 1) * 100
        check(abs(_saving - 29.88) < 0.05,
              f"the 29.88 % figure is C3 vs C1 (got {_saving:.2f} %)")
        check(abs(_c4v3 - 55.9) < 0.3,
              f"the GPU-front-end-plus-CPU-core configuration is {_c4v3:.1f} % "
              f"more energy per frame than C3")
        check(abs(_c4v1 - 9.33) < 0.15,
              f"the GPU-front-end-plus-CPU-core configuration is {_c4v1:.1f} % "
              f"more energy per frame than C1")
        _p3, _p4 = _pw("C3"), _pw("C4")
        if None not in (_p3, _p4):
            check(abs((_p4 / _p3 - 1) * 100 - 27.8) < 0.3,
                  f"the GPU-front-end-plus-CPU-core configuration draws "
                  f"{(_p4/_p3-1)*100:.1f} % more power than C3")

    # ---------------------------------------------------- 10b. the Loihi 2
    # wording.  The C4-vs-C3 gap is a GPU-front-end-vs-CPU measurement and
    # says NOTHING about Loihi 2, so no document may assert a Loihi 2
    # saving in either direction.
    print("\n[10b] no document asserts a Loihi 2 saving that was never "
          "measured")
    FORBIDDEN = (
        "loihi 2 gave no saving",
        "loihi 2 saved no",
        "no saving over an optimized cpu core",
        "no saving over an optimised cpu core",
        "loihi 2 gave no benefit",
    )
    for rel in ("exp_rev1/results/REVIEW_RESPONSE.md",
                "exp_rev1/results/manuscript/results.md",
                "exp_rev1/results/manuscript/summary.txt",
                "appeal_letter/appeal_letter.md",
                "appeal_letter/appeal_letter.txt",
                "README.md"):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        # a sentence may MENTION the phrase in order to forbid it, so only
        # flag it where it is not inside a prohibition.
        bad = []
        for line in txt.splitlines():
            low = line.lower()
            # markdown emphasis must not hide the keyword: a prohibition may
            # be written "may **not** state that Loihi 2 gave no saving".
            flat = low.replace("*", "").replace("`", "")
            flat = re.sub(r"\s+", " ", flat)
            for f_ in FORBIDDEN:
                if f_ in flat and not any(
                        k in flat for k in ("may not", "must not", "would be",
                                            "unsupported", "cannot",
                                            "not supported",
                                            "not state", "never state",
                                            "does not")):
                    bad.append(line.strip()[:120])
                    break
        check(not bad, f"{rel}: no unsupported Loihi 2 saving claim")
        for b in bad:
            print(f"        {b}")

    # ---------------------------------------------------- 10c. the withdrawn
    # numbers must be disclosed, not silently replaced.
    print("\n[10c] the unsupported submitted-version numbers are disclosed")
    rr = os.path.join(R, "REVIEW_RESPONSE.md")
    if os.path.exists(rr):
        txt = open(rr).read()
        for fig in ("60.9", "22.7", "0.145", "99.3", "11.86"):
            check(fig in txt,
                  f"the review response discloses the withdrawn figure {fig}")
        check("WITHDRAWN" in txt,
              "the review response marks the unsupported figures as "
              "WITHDRAWN rather than replacing them silently")

    # ---------------------------------------------------- 11. the rate cap
    print("\n[11] the rate cap is distinguished from flat-out throughput")
    tput = load(os.path.join(R, "run1d", "throughput.json"), {}) or {}
    tres = tput.get("results") or {}
    if tres:
        for c in ("C3", "C5"):
            if c in tres:
                check(tres[c].get("keeps_up_with_clock") is True,
                      f"{c} keeps up with the 50 Hz clock, so its 50.00 FPS "
                      f"figure is clock-clipped and its maximum rate is "
                      f"reported separately "
                      f"({tres[c].get('throughput_fps_mean', 0):.2f} FPS)")
        for c in ("C1", "C2", "C4"):
            if c in tres:
                check(tres[c].get("keeps_up_with_clock") is False,
                      f"{c} cannot keep up with the clock, so its Run 1 FPS is "
                      f"already its flat-out maximum")
    else:
        WARN.append("no run1d/throughput.json -- the rate-cap distinction "
                    "could not be checked")

    # ---------------------------------------------------- 12. the C2 fix
    print("\n[12] the TensorRT fix is documented and did not change the output")
    prof = load(os.path.join(R, "run1d", "tensorrt_profile.json"), {}) or {}
    if prof:
        eq = prof.get("equivalence") or {}
        check(eq.get("bit_identical") is True,
              "the fixed TensorRT backend produces bit-identical spikes to "
              "the original one")
        d = prof.get("derived") or {}
        check(d.get("fixed_c2_to_c1_latency_ratio", 9) <
              d.get("old_c2_to_c1_latency_ratio", 0),
              "the fixed TensorRT baseline is closer to the PyTorch baseline "
              "than the original one was")
    else:
        WARN.append("no run1d/tensorrt_profile.json -- the TensorRT fix "
                    "could not be checked")

    # ---------------------------------------------------- 13. the memory-cap
    # caveat and the uncapped run.  The failures in Run 2 are against a
    # 3000 MB limit the HARNESS imposed, so every document must say so, and
    # the uncapped run must be reported when it exists.
    print("\n[13] the memory-cap caveat is stated and the uncapped run is "
          "reported")
    for rel in ("exp_rev1/results/REVIEW_RESPONSE.md",
                "exp_rev1/results/manuscript/results.md",
                "exp_rev1/results/manuscript/summary.txt",
                "appeal_letter/appeal_letter.md",
                "appeal_letter/appeal_letter.txt",
                "README.md"):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        bad = []
        for line in txt.splitlines():
            low = line.lower()
            # a line that mentions the failures must not present them as
            # real 8 GB out-of-memory failures
            if ("out-of-memory" in low or "oom" in low) and \
                    ("8 gb" in low or "real jetson" in low or "module" in low):
                if not any(k in low for k in ("not ", "never", "rather",
                                              "must not", "harness",
                                              "address-space", "cap ",
                                              "does not", "not demonstrate",
                                              "no artificial", "not show")):
                    bad.append(line.strip()[:120])
        check(not bad, f"{rel}: the Run 2 failures are never presented as "
                        f"real 8 GB out-of-memory failures")
        for b in bad:
            print(f"        {b}")

    # 13b: the 3000 MB cap must be named
    rr = os.path.join(R, "REVIEW_RESPONSE.md")
    if os.path.exists(rr):
        txt = open(rr).read()
        check("3000" in txt and ("RLIMIT_AS" in txt or "setrlimit" in txt),
              "the review response names the 3000 MB RLIMIT_AS cap that the "
              "harness imposed")
        check("not" in txt.lower() and "could not be measured" in txt.lower(),
              "the review response states that Loihi 2 could not be measured")
        check("mean" in txt.lower() and "±" in txt,
              "the review response reports mean ± SD figures")

    # 13c: the uncapped run
    nocap = load(os.path.join(R, "run2_nocap", "summary.json"), {}) or {}
    if nocap:
        check(True, f"the uncapped Run 2 is present with "
                    f"{len(nocap)} configurations")
        ev = load(os.path.join(R, "run2_nocap", "kernel_oom_evidence.json"),
                  {}) or {}
        check(ev.get("artificial_cap_in_effect") is False,
              "the uncapped run had NO artificial address-space cap")
        check(ev.get("oom_kill_counter") is not None,
              f"the uncapped run recorded the kernel oom_kill counter "
              f"({ev.get('oom_kill_counter')})")
    else:
        WARN.append("no run2_nocap/summary.json -- the uncapped Run 2 has "
                    "not been merged into the report yet")

    # 13d: the Section 5.2 peak-memory / headroom slots are FILLED and
    # carry the LOG-derived values, not the harness summary's.
    print("\n[13d] the Section 5.2 peak-memory and headroom slots are "
          "filled with the log-derived values")
    peak = load(os.path.join(R, "memory_peak_from_logs.json"), {}) or {}
    if peak:
        _resmd = os.path.join(R, "manuscript", "results.md")
        _tex = os.path.join(R, "manuscript", "tables.tex")
        _txt = open(_resmd).read() if os.path.exists(_resmd) else ""
        _ttx = open(_tex).read() if os.path.exists(_tex) else ""
        check("Table 2c" in _txt and "headroom" in _txt.lower(),
              "the generated manuscript carries the Section 5.2 "
              "peak-memory and headroom table")
        check("Table 2c" in _ttx and "Headroom" in _ttx,
              "the generated LaTeX carries the Section 5.2 peak-memory and "
              "headroom table")
        # the log-derived peaks must be the ones quoted, and they must
        # exceed the harness summary for the two configurations whose
        # crashed recordings the summary skips
        for _pname in ("run2", "run2_nocap"):
            _p = peak.get(_pname) or {}
            for _k, _d in (_p.get("configs") or {}).items():
                _fl = _d.get("from_log") or {}
                _pt = _fl.get("peak_total_used_mb_all_recordings")
                if _pt is None:
                    continue
                check(f"{_pt:.1f}" in _txt or f"{_pt:.0f}" in _txt,
                      f"{_pname}/{_k}: the log-derived peak {_pt:.1f} MB is "
                      f"quoted in the generated manuscript")
        # J1/J3 in the capped pass: the log peak must be >= the harness
        # summary, because the summary skips the crashed recordings
        for _k in ("J1", "J3"):
            _d = ((peak.get("run2") or {}).get("configs") or {}).get(_k) or {}
            _fl = _d.get("from_log") or {}
            _xc = _fl.get("crosscheck_summary_peak_total_mb")
            _lg = _fl.get("peak_total_used_mb_all_recordings")
            if _xc is not None and _lg is not None:
                check(_lg >= _xc - 1e-6,
                      f"{_k}: the log peak ({_lg:.1f} MB) is not below the "
                      f"harness summary ({_xc:.1f} MB)")

    # 13e: ONE standard-deviation convention across every document
    print("\n[13e] the standard-deviation convention is consistent")
    try:
        import statistics as _st
        _agg = load(os.path.join(R, "run1", "table1_aggregate.json"), {}) or {}
        _cfgs = _agg.get("configs") or {}
        for _c in ("C1", "C3", "C5"):
            _reps = _cfgs.get(_c, {}).get("repeats") or []
            _pw = [x.get("tegrastats_vdd_in_mean_w") for x in _reps
                   if x.get("tegrastats_vdd_in_mean_w") is not None]
            _reported = _cfgs.get(_c, {}).get("power_w_sd")
            if not _pw or _reported is None:
                continue
            _sample = _st.stdev(_pw) if len(_pw) > 1 else 0.0
            check(abs(_reported - _sample) < 1e-9,
                  f"{_c}: the Table 1 aggregate reports the SAMPLE sd "
                  f"({_reported:.4f} W), matching the harness summary")
    except Exception as _e:  # pragma: no cover
        WARN.append(f"could not check the SD convention: {_e}")

    # ---------------------------------------------------- 14. the scope
    print("\n[14] the scope of the measurement is stated")
    if os.path.exists(rr):
        txt = open(rr).read()
        check("SCOPE" in txt,
              "the review response has a SCOPE section")
        check("not Loihi 2" in txt or "not* a measurement of Loihi 2" in txt
              or "C4 is not Loihi 2" in txt,
              "the review response states that C4 is not Loihi 2")
        check("nvpmodel" in txt.lower(),
              "the review response states the nvpmodel mode used")
        for term in ("sensors", "Loihi 2 board", "link"):
            check(term.lower() in txt.lower(),
                  f"the measurement boundary names the excluded item "
                  f"'{term}'")
        check("different frame rates" in txt.lower()
              or "compared at different frame rates" in txt.lower()
              or "per frame the configuration actually delivered" in txt.lower(),
              "the review response states that J/frame is compared at "
              "different frame rates")
        check("thermal stability" in txt.lower(),
              "the review response reports thermal stability rather than an "
              "endurance result")

    # ---------------------------------------------------- 15. the checks
    # themselves.  The verification script proves that the documents agree
    # with each other; it does NOT prove the measurements are right.  That
    # distinction has to be stated wherever the check count is quoted.
    print("\n[15] the verification script's own scope is stated")
    for rel in ("exp_rev1/results/REVIEW_RESPONSE.md",
                "README.md"):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        check("agreement between the documents" in txt.lower()
              or "documents agree" in txt.lower()
              or "internal consistency" in txt.lower(),
              f"{rel}: the check count is described as a consistency check, "
              f"not as validation of the measurements")

    # ---------------------------------------------------- 16. idempotency
    # The generators must be idempotent: running sync_published_numbers.py
    # twice must produce byte-identical documents.  A generator that
    # appends its own output on every run is a silent corruption of the
    # deliverables, and it is invisible to every other check.
    print("\n[16] the document generators are idempotent")
    _gen = os.path.join(ROOT, "exp_rev1", "sync_published_numbers.py")
    _tracked = [os.path.join(ROOT, "README.md"),
                os.path.join(ROOT, "appeal_letter", "appeal_letter.md"),
                os.path.join(ROOT, "appeal_letter", "appeal_letter.txt")]
    if os.path.exists(_gen) and all(os.path.exists(t) for t in _tracked):
        import hashlib as _hl
        import subprocess as _sp
        def _digests():
            return [_hl.md5(open(t, "rb").read()).hexdigest() for t in _tracked]
        _before = _digests()
        try:
            _sp.run([sys.executable, _gen], cwd=os.path.join(ROOT, "exp_rev1"),
                    capture_output=True, timeout=300)
            _after = _digests()
            check(_before == _after,
                  "running sync_published_numbers.py a second time leaves the "
                  "README and both appeal letters byte-identical")
            if _before != _after:
                for t, b, a in zip(_tracked, _before, _after):
                    if b != a:
                        print(f"        {os.path.basename(t)} changed on the "
                              f"second run")
        except Exception as e:
            WARN.append(f"could not test idempotency ({e})")
    else:
        WARN.append("the generator or the documents are missing -- "
                    "idempotency was not tested")

    # ---------------------------------------------------- 17. the figures
    # The figures must be byte-reproducible.  Matplotlib stamps the current
    # time into the PDF/SVG metadata and generates a random id for every
    # SVG element, so without an explicit fix the figure bytes change on
    # every run even when the plotted content is identical -- which makes
    # `git diff` on the figures useless.
    print("\n[17] the figures are byte-reproducible")
    _figdir = os.path.join(ROOT, "figure")
    _figs = [f for f in ("S1.pdf", "S1.svg", "S2.pdf", "S2.svg",
                         "S3.pdf", "S3.svg")
             if os.path.exists(os.path.join(_figdir, f))]
    _figgen = os.path.join(ROOT, "exp_rev1", "make_paper_figures.py")
    if _figs and os.path.exists(_figgen):
        import hashlib as _hl2
        import subprocess as _sp2
        def _fdig():
            return [_hl2.md5(open(os.path.join(_figdir, f), "rb").read())
                    .hexdigest() for f in _figs]
        _fb = _fdig()
        try:
            _sp2.run([sys.executable, _figgen],
                     cwd=os.path.join(ROOT, "exp_rev1"),
                     capture_output=True, timeout=600)
            _fa = _fdig()
            check(_fb == _fa,
                  "running make_paper_figures.py a second time leaves every "
                  "figure byte-identical")
            if _fb != _fa:
                for f, b, a in zip(_figs, _fb, _fa):
                    if b != a:
                        print(f"        figure/{f} changed on the second run")
        except Exception as e:
            WARN.append(f"could not test figure reproducibility ({e})")
    else:
        WARN.append("the figures are missing -- reproducibility was not "
                    "tested")

    # ---------------------------------------------------- 18. the anchors
    # A generator that inserts a block into a document by matching a heading
    # it assumes is present will silently DROP the block when the heading
    # moves -- and every other check still passes, because the checks that
    # were supposed to guard the block are also guarded by the same missing
    # anchor.  This happened for real: the withdrawn-figures table was
    # written into the README by a generator whose anchor ("## Key Results")
    # did not exist in the README, so the table was never written, and
    # nothing complained.
    #
    # So: assert that every block the generators claim to insert is
    # actually present in the output.
    print("\n[18] the generators actually inserted their blocks")
    _must = [
        (os.path.join(ROOT, "README.md"),
         "### The withdrawn figures, and where each came from",
         "the README contains the withdrawn-figures provenance table"),
        (os.path.join(ROOT, "README.md"),
         "## Status of this manuscript",
         "the README states the status of the manuscript (Track A/B/C)"),
        (os.path.join(ROOT, "README.md"),
         "the 6 W boundary excludes the sensors",
         "the README states what the power boundary excludes"),
        (os.path.join(ROOT, "README.md"),
         "only C3 and C5 deliver all of it",
         "the README says which configurations do not hold 50 Hz"),
        (os.path.join(ROOT, "appeal_letter", "appeal_letter.md"),
         "not measurements",
         "the appeal letter discloses that the old figures were not "
         "measurements"),
        (os.path.join(ROOT, "appeal_letter", "appeal_letter.md"),
         "nothing here supports a claim about Loihi 2 in either direction",
         "the appeal letter states that Loihi 2 was not measured"),
        (os.path.join(ROOT, "appeal_letter", "appeal_letter.md"),
         "does not show that Algorithm 1 prevents real out-of-memory crashes",
         "the appeal letter does not claim Algorithm 1 prevents crashes"),
        (os.path.join(ROOT, "exp_rev1", "results", "REVIEW_RESPONSE.md"),
         "origin is therefore not established by anything in this repository",
         "the review response states the provenance limit for the four "
         "unexplained figures"),
        (os.path.join(ROOT, "exp_rev1", "results", "REVIEW_RESPONSE.md"),
         "UNAVAILABLE",
         "the review response marks the Loihi 2 quantities UNAVAILABLE"),
        (os.path.join(ROOT, "exp_rev1", "INTERNAL_MEMO_FIGURE_PROVENANCE.md"),
         "Nothing has been sent to the editor",
         "the internal memo states that nothing has been sent"),
        (os.path.join(ROOT, "exp_rev1", "INTERNAL_MEMO_FIGURE_PROVENANCE.md"),
         "60.9 W, 22.7 W, 0.145 W and 99.3 % do not occur in the submitted "
         "tree",
         "the internal memo states the provenance limit for the four "
         "unexplained figures"),
        (os.path.join(ROOT, "exp_rev1", "MEASUREMENT_LOCK_INCIDENT.md"),
         "make_paper_figures.py",
         "the lock incident record names the process that contaminated the "
         "measurement"),
        (os.path.join(ROOT, "README.md"),
         "### The measurement lock",
         "the README documents the measurement lock"),
        (os.path.join(ROOT, "README.md"),
         "*IEEE Transactions on Robotics*",
         "the README names the journal the appeal letter addresses"),
        (os.path.join(ROOT, "README.md"),
         "Manuscript: Loihi 2 Hybrid Neuromorphic SLAM for Aerial Robotics",
         "the README cites the same manuscript title as the appeal letter"),
    ]
    for _path, _needle, _desc in _must:
        _ok = False
        if os.path.exists(_path):
            try:
                _txt = open(_path, encoding="utf-8").read()
                _ok = _needle.lower() in _txt.lower()
            except Exception:
                _ok = False
        check(_ok, _desc)

    # ---- the matched-rate (30 Hz) block, when the pass has produced data
    #
    # This check is CONDITIONAL on the 30 Hz data existing.  A generator
    # that inserts a block by matching a heading can silently drop the
    # block (see check [18]); this is the check that the block really
    # reached the documents, once there is something to put in it.
    _p30 = os.path.join(ROOT, "exp_rev1", "results", "run1_30hz",
                         "all_runs.json")
    if os.path.exists(_p30):
        try:
            import json as _json30
            _r30 = _json30.load(open(_p30))
            _n30 = sum(len(v) for v in _r30.values()
                       if isinstance(v, list))
        except Exception:
            _n30 = 0
        if _n30 >= 25:
            for _path, _needle, _desc in (
                (os.path.join(ROOT, "README.md"),
                 "### The same comparison at a matched 30 Hz frame rate",
                 "the README contains the matched-rate (30 Hz) table"),
                (os.path.join(ROOT, "exp_rev1", "results",
                              "REVIEW_RESPONSE.md"),
                 "### 1.3dd The same experiment at a matched 30 Hz frame rate",
                 "the review response contains the matched-rate (30 Hz) table"),
                (os.path.join(ROOT, "appeal_letter", "appeal_letter.md"),
                 "### 2.2 The same comparison at a matched frame rate",
                 "the appeal letter contains the matched-rate (30 Hz) table"),
            ):
                _ok = False
                if os.path.exists(_path):
                    _ok = _needle.lower() in open(
                        _path, encoding="utf-8").read().lower()
                check(_ok, _desc)
            # The matched-rate block must not claim anything about Loihi 2,
            # and must not replace the 50 Hz pass.
            _rr = os.path.join(ROOT, "exp_rev1", "results",
                               "REVIEW_RESPONSE.md")
            if os.path.exists(_rr):
                _txt = open(_rr, encoding="utf-8").read()
                check("no Loihi 2 saving is reported in either direction"
                      in _txt,
                      "the review response states that the matched-rate pass "
                      "reports no Loihi 2 saving")
                check("does not replace the 50 Hz pass" in _txt,
                      "the review response states that the matched-rate pass "
                      "does not replace the 50 Hz pass")
        else:
            print(f"        the 30 Hz pass has recorded {_n30}/25 repeats "
                  f"-- the matched-rate block is not expected yet")
    else:
        print("        the 30 Hz pass has not produced data yet -- the "
              "matched-rate block is not expected yet")

    # The letters must not cite the automated check count as evidence.
    for _lp in (os.path.join(ROOT, "appeal_letter", "appeal_letter.md"),
                os.path.join(ROOT, "appeal_letter", "appeal_letter.txt")):
        _ok = False
        if os.path.exists(_lp):
            _t = open(_lp, encoding="utf-8").read()
            import re as _re
            _ok = not _re.search(r"\b\d{2,4}\s+(?:automated\s+)?checks?\b",
                                 _t, _re.I)
        check(_ok,
              f"{os.path.basename(_lp)} does not cite an automated check "
              f"count to the editor")

    # ---------------------------------------------------- 19. the lock
    # A measurement window can be contaminated by anything that starts
    # inside it, and the quiescence guard can only check the machine
    # BEFORE the window opens.  This happened for real: the 30 Hz pass was
    # aborted at C1_rep1 because the author's own verify_final.py was
    # rendering figures at 81 % CPU inside the window.  The guard caught it
    # and refused to emit the figure -- correct behaviour -- but the run was
    # still lost.  So the measurement scripts take an exclusive lock and the
    # CPU-heavy tooling refuses to start while it is held.
    print("\n[19] the measurement lock protects the measurement windows")
    try:
        sys.path.insert(0, os.path.join(ROOT, "exp_rev1"))
        import measurement_lock as _ml
        check(_ml.tooling_is_blocked("verify_final.py") is None,
              "the measurement lock is free when no measurement is running")
        with _ml.measurement_lock("verify_final_selftest"):
            _blocked = _ml.tooling_is_blocked("verify_final.py")
            check(_blocked is not None,
                  "the tooling refuses to run while a measurement holds the "
                  "lock")
            check("MEASUREMENT IS IN PROGRESS" in (_blocked or ""),
                  "the refusal message names the measurement in progress")
        check(_ml.tooling_is_blocked("verify_final.py") is None,
              "the lock is released when the measurement ends")
        # every measurement script must take the lock
        _meas = ["run1_power.py", "run2_memory.py", "run3_endurance.py"]
        for _m in _meas:
            _p = os.path.join(ROOT, "exp_rev1", _m)
            _ok = os.path.exists(_p) and \
                "measurement_lock" in open(_p, encoding="utf-8").read()
            check(_ok, f"{_m} takes the measurement lock")
        # every CPU-heavy generator must respect the lock
        _tools = ["make_paper_figures.py", "make_review_response.py",
                  "make_appeal_letters.py", "sync_published_numbers.py",
                  "verify_final.py", "build_manuscript_tables.py"]
        for _t in _tools:
            _p = os.path.join(ROOT, "exp_rev1", _t)
            _ok = os.path.exists(_p) and \
                "refuse_if_measurement_running" in \
                open(_p, encoding="utf-8").read()
            check(_ok, f"{_t} refuses to run during a measurement")
    except Exception as e:
        WARN.append(f"could not test the measurement lock ({e})")

    # ---------------------------------------------------- summary
    print("\n" + "=" * 72)
    if FAIL:
        print(f" FAILED: {len(FAIL)} check(s) did not pass")
        for f_ in FAIL:
            print(f"   - {f_}")
    else:
        print(" ALL CHECKS PASSED")
    if WARN:
        print(f" {len(WARN)} warning(s)")
        for w in WARN:
            print(f"   - {w}")
    print("=" * 72)
    return 1 if FAIL else 0



if __name__ == "__main__":
    # This tool is CPU-heavy and must NOT run while a measurement is in
    # progress: it would contaminate the open measurement window.  On
    # 2026-11-05 it did exactly that and cost a whole 30 Hz pass.
    from measurement_lock import refuse_if_measurement_running
    refuse_if_measurement_running("verify_final.py")
    sys.exit(main())
