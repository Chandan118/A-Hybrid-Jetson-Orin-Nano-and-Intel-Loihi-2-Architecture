#!/usr/bin/env python3
"""S1/S8 single-process Lava equivalence (runs INSIDE the lava venv).

Why this exists: the parent/child socket protocol (_test_c6.py +
lava_c6_runner.py) stalls on this bench -- both Lava children compiled
and initialized, then died with EOF waiting for the parent's first
command while both parents hung (two runtimes were also competing).
This script avoids that protocol entirely and drives the Runtime
directly in one process, exactly like _probe_c6.py but extended to
N frames with the threshold-tie rule from test_backends_equivalent.py.

Usage:  /tmp/lavavenv/bin/python -u s1_singleproc.py [exact|lava] [nframes]
Writes: results/s1_<variant>_singleproc.json
"""
import os
import sys
import json
import shutil
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lava.magma.compiler.compiler import Compiler
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.magma.runtime.runtime import Runtime
from lava.magma.core.process.message_interface_enum import ActorType

import workload as WL
import lava_c6_model

TIE_MARGIN = 0.05


def reference(x, W, cfg):
    beta, thr, T = cfg["beta"], cfg["spike_threshold"], cfg["time_steps"]
    w1, w2, w3 = (W[k].astype(np.float64) for k in ("w1", "w2", "w3"))
    m1 = np.zeros(cfg["hidden_size"])
    m2 = np.zeros(cfg["hidden_size"])
    m3 = np.zeros(cfg["output_size"])
    x = np.asarray(x, dtype=np.float64)
    out = None
    margin = np.full(cfg["output_size"], np.inf, dtype=np.float64)
    hidden_min = np.inf
    for _ in range(T):
        m1 = beta * m1 + (x @ w1)
        s1 = (m1 >= thr).astype(np.float64)
        hidden_min = min(hidden_min, float(np.abs(m1 - thr).min()))
        m1 = m1 - thr * s1
        m2 = beta * m2 + (s1 @ w2)
        s2 = (m2 >= thr).astype(np.float64)
        hidden_min = min(hidden_min, float(np.abs(m2 - thr).min()))
        m2 = m2 - thr * s2
        m3 = beta * m3 + (s2 @ w3)
        out = (m3 >= thr).astype(np.float64)
        margin = np.minimum(margin, np.abs(m3 - thr))
        m3 = m3 - thr * out
    margin = np.minimum(margin, hidden_min)
    return out, margin, hidden_min


def zero_state(rt, lifs):
    for lf in lifs:
        for var in (getattr(lf, "v", None), getattr(lf, "u", None)):
            if var is None:
                continue
            print(f"  zero {lf.name} var size {int(np.prod(var.shape))}",
                  flush=True)
            rt.set_var(var.id, np.zeros(int(np.prod(var.shape)),
                                        dtype=np.float32))
    print("  state zeroed", flush=True)


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "exact"
    nframes = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    t_all = time.perf_counter()
    W = WL.load_weights(os.path.join(HERE, "weights_quantised.npz"))
    for bk in ("b1", "b2", "b3"):
        assert np.all(W[bk] == 0), "bias must be zero for Lava Dense"
    cfg = dict(WL.NET)

    workdir = tempfile.mkdtemp(prefix="lava_s1_")
    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        src, rec, denses, lifs = lava_c6_model.build_network(
            W, cfg, variant=variant)
        print("compiling ...", flush=True)
        exe = Compiler().compile(src, Loihi2SimCfg())
        print("COMPILED OK", flush=True)
    finally:
        os.chdir(cwd)
        shutil.rmtree(workdir, ignore_errors=True)

    rt = Runtime(exe, ActorType.MultiProcessing)
    rt.initialize()
    print("INITIALIZED", flush=True)
    rt.start(RunSteps(num_steps=1, blocking=True))
    print("FIRST STEP DONE", flush=True)

    scans = WL.read_scans(max_scans=nframes)
    T = cfg["time_steps"]
    frames = []
    tot_bad = tot_nontie = tot_units = tot_tie = 0
    try:
        for i in range(len(scans)):
            x = WL.encode_input(WL.profile_from_scan(scans[i][1]))
            print(f"frame {i}: zero_state ...", flush=True)
            zero_state(rt, lifs)
            print(f"frame {i}: set frame, run {T} steps ...", flush=True)
            t0 = time.perf_counter()
            rt.set_var(src.frame.id, np.asarray(x, dtype=np.float32))
            rt.start(RunSteps(num_steps=T, blocking=True))
            last = np.asarray(rt.get_var(rec.last.id)).copy()
            dt = (time.perf_counter() - t0) * 1000.0
            ref, margin, hmin = reference(x, W, cfg)
            lava = (last > 0)
            bad = (lava.astype(float) != ref)
            tie = margin < TIE_MARGIN
            nontie = bad & (~tie)
            tot_bad += int(bad.sum()); tot_nontie += int(nontie.sum())
            tot_units += ref.size; tot_tie += int(tie.sum())
            frames.append({"frame": i, "wall_ms": dt,
                           "agree": int((~bad).sum()), "units": int(ref.size),
                           "lava_spikes": int(lava.sum()),
                           "ref_spikes": int(ref.sum()),
                           "tie_units": int(tie.sum()),
                           "nontie_bad": int(nontie.sum()),
                           "hidden_min": hmin})
            print(f"frame {i}: wall={dt:.0f}ms agree={int((~bad).sum())}/64 "
                  f"lava_spk={int(lava.sum())} ref_spk={int(ref.sum())} "
                  f"nontie_bad={int(nontie.sum())}", flush=True)
    finally:
        try:
            rt.stop()
        except Exception as e:
            print(f"rt.stop: {e}", flush=True)

    ok = (tot_nontie == 0) if variant == "exact" else None
    out = {"variant": variant, "n_frames": len(frames),
           "weights": "weights_quantised.npz", "bag": WL.DEFAULT_BAG,
           "seed": 12345, "lava_nc": "0.10.0", "tie_margin": TIE_MARGIN,
           "total_units": tot_units, "total_bad": tot_bad,
           "total_nontie_bad": tot_nontie, "total_tie_units": tot_tie,
           "verdict": ("PASS" if ok else "FAIL") if ok is not None
                      else "REPORTED (deviation expected)",
           "wall_s": time.perf_counter() - t_all, "frames": frames}
    p = os.path.join(os.path.dirname(HERE), "results",
                     f"s1_{variant}_singleproc.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"verdict={out['verdict']} bad={tot_bad}/{tot_units} "
          f"nontie_bad={tot_nontie} -> {p}", flush=True)


if __name__ == "__main__":
    main()
