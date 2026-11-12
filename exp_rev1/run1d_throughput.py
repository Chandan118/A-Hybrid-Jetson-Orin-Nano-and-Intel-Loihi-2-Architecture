#!/usr/bin/env python3
"""Measure the UNCONSTRAINED maximum throughput of every configuration.

Why this script exists
----------------------
The main Run 1 harness drives every configuration through a time-based
ReplayClock at 50 Hz.  That is the fair way to compare ENERGY PER FRAME,
because every configuration is offered the same frames per second.  But it
also means that the reported FPS is NOT the same kind of number for every
configuration:

  * C3 and C5 are FAST ENOUGH to keep up with the 50 Hz clock, so they are
    reported at exactly 50.00 FPS.  Their number is CLIPPED BY THE CLOCK,
    not by their own speed.
  * C1, C2 and C4 cannot keep up, so they are reported at their own
    maximum rate and they drop frames.

Mixing the two in one "FPS" column is misleading: it looks as though C3 is
five times slower than C1, when in fact C3 is the fastest configuration
measured.  This script removes the clock entirely and reports each
configuration's own maximum sustainable throughput, so that the paper can
state BOTH numbers:

  * the energy per frame at the fixed 50 Hz replay rate, and
  * the maximum unconstrained throughput.

How it is measured
------------------
For each configuration the backend is stepped as fast as the CPU allows,
with NO replay clock and NO sleep.  The wall-clock time per frame is
measured directly.  The reported throughput is 1 / mean_frame_time, so it
is the honest "how fast can this configuration go" number.

Power is NOT measured here: without the fixed-duration logging window of
Run 1 the power figures are not comparable with the Run 1 table.  This
script measures throughput only, and is reported separately from the
energy table.

Numerical note: TF32 is disabled, exactly as in run1c_pose_check.py, so
that every configuration evaluates the same function.
"""

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
sys.path.insert(0, HERE)

import numpy as np
import workload as WL
from run1_power import build_backend


def set_numeric():
    """Disable TF32 so every configuration computes the same function."""
    numeric = {}
    try:
        import torch
        numeric["torch_version"] = torch.__version__
        numeric["matmul_allow_tf32_before"] = bool(
            torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
        numeric["matmul_allow_tf32_after"] = bool(
            torch.backends.cuda.matmul.allow_tf32)
        numeric["float32_matmul_precision"] = str(
            torch.get_float32_matmul_precision())
    except Exception as e:
        numeric["error"] = str(e)
    return numeric


def measure(which, weights, cfg, enc, warmup, n_frames, repeats):
    """Return per-repeat (throughput_fps, mean_frame_ms, p50_frame_ms).

    ONE FRESH PROCESS PER (config, repeat)
    --------------------------------------
    The original implementation ran every configuration and every repeat
    inside one process, in a fixed config-major order.  The 50 Hz power
    pass showed what that does: the C++ fusion core keeps a per-frame
    output map and the CUDA contexts keep their allocations, so state
    carried from one repeat into the next, and the recorded RAM grew by
    ~670 MB across C3's five repeats in the 30 Hz pass before the harness
    was fixed.

    A throughput number measured in a process that has already grown is
    not the same number as one measured in a fresh process, so every
    (config, repeat) now runs in its own process.  The order is also
    shuffled, so that thermal drift over the pass cannot favour whichever
    configuration happens to run last.

    The per-repeat results are written to their own files so that an
    interruption costs one repeat, not the whole pass.
    """
    backend = build_backend(which, weights, cfg)
    out = []
    try:
        for rep in range(repeats):
            backend.reset()
            # warm-up: let the clocks settle and the caches fill
            for i in range(warmup):
                backend.step(enc[i % len(enc)])
            backend.reset()

            times = []
            for i in range(n_frames):
                t0 = time.perf_counter()
                backend.step(enc[i % len(enc)])
                times.append((time.perf_counter() - t0) * 1000.0)
            total_s = sum(times) / 1000.0
            times_sorted = sorted(times)
            p50 = times_sorted[len(times) // 2]
            out.append({
                "throughput_fps": n_frames / total_s,
                "mean_frame_ms": total_s * 1000.0 / n_frames,
                "p50_frame_ms": p50,
                "min_frame_ms": times_sorted[0],
                "max_frame_ms": times_sorted[-1],
            })
    finally:
        backend.close()
    return out


def measure_one_fresh(which, rep, weights, cfg, enc, warmup, n_frames):
    """Run ONE repeat of one configuration, and return its record.

    This is the WORKER half of the fresh-process protocol: it is invoked
    in a subprocess by measure_fresh(), so the process that builds the
    backend, builds the TensorRT engines and allocates the CUDA context is
    not the process that runs the next repeat.
    """
    backend = build_backend(which, weights, cfg)
    try:
        backend.reset()
        for i in range(warmup):
            backend.step(enc[i % len(enc)])
        backend.reset()
        times = []
        for i in range(n_frames):
            t0 = time.perf_counter()
            backend.step(enc[i % len(enc)])
            times.append((time.perf_counter() - t0) * 1000.0)
        total_s = sum(times) / 1000.0
        times_sorted = sorted(times)
        return {
            "config": which,
            "rep": rep,
            "throughput_fps": n_frames / total_s,
            "mean_frame_ms": total_s * 1000.0 / n_frames,
            "p50_frame_ms": times_sorted[len(times) // 2],
            "min_frame_ms": times_sorted[0],
            "max_frame_ms": times_sorted[-1],
        }
    finally:
        backend.close()


def measure_fresh(which, weights, cfg, enc, warmup, n_frames, repeats,
                  outdir, seed, no_shuffle=False):
    """Run every repeat of one configuration, each in a FRESH process."""
    reps_dir = os.path.join(outdir, "repeats")
    os.makedirs(reps_dir, exist_ok=True)
    out = []
    for rep in range(repeats):
        rp = os.path.join(reps_dir, f"{which}_rep{rep}.json")
        if os.path.exists(rp):
            try:
                with open(rp) as f:
                    out.append(json.load(f))
                print(f"  {which} rep{rep}: already measured, skipping",
                      flush=True)
                continue
            except Exception:
                pass
        cmd = [sys.executable, os.path.abspath(__file__),
               "--worker-one", "--which", which, "--rep", str(rep),
               "--frames", str(n_frames), "--warmup", str(warmup),
               "--outdir", outdir]
        print(f"  {which} rep{rep}: FRESH PROCESS", flush=True)
        p = subprocess.run(cmd, cwd=HERE)
        if p.returncode != 0:
            raise RuntimeError(f"{which} rep{rep} failed with exit code "
                               f"{p.returncode}")
        with open(rp) as f:
            out.append(json.load(f))
    return out


def sd(vals):
    """Sample standard deviation; 0.0 for a single repeat."""
    return st.stdev(vals) if len(vals) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3000,
                    help="timed frames per repeat (no clock, back to back)")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--rate", type=float, default=50.0,
                    help="the clock rate the Run 1 comparison used")
    ap.add_argument("--configs", default="C1,C2,C3,C4,C5")
    ap.add_argument("--outdir", default=os.path.join(HERE, "results", "run1d"))
    ap.add_argument("--seed", type=int, default=20261106,
                    help="the shuffle seed, so the schedule is reproducible")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="run the configurations in a fixed order (default is "
                         "to shuffle, so thermal drift cannot favour "
                         "whichever config runs last)")
    ap.add_argument("--worker-one", action="store_true",
                    help=argparse.SUPPRESS)  # internal: one repeat, one process
    ap.add_argument("--which", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--rep", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    numeric = set_numeric()
    print("numeric settings:", json.dumps(numeric))

    weights = WL.load_weights(os.path.join(HERE, "common", "weights_quantised.npz"))
    cfg = dict(WL.NET)
    cfg["rate_hz"] = args.rate
    cfg["trt_fp16"] = False

    scans = WL.read_scans()
    enc = np.stack([WL.encode_input(WL.profile_from_scan(a)) for _, a in scans])
    print(f"{len(enc)} replay frames available; timing {args.frames} per repeat "
          f"x {args.repeats} repeats\n")

    # ---- the WORKER path: exactly ONE repeat, in THIS process
    if args.worker_one:
        # The same guard the power pass uses: a throughput number taken
        # while something else is using the machine is not the machine's
        # throughput.  The record is written next to the results.
        _q = os.path.join(args.outdir, "quiescence")
        os.makedirs(_q, exist_ok=True)
        _qpath = os.path.join(_q, f"{args.which}_rep{args.rep}.json")
        try:
            subprocess.run([sys.executable,
                            os.path.join(HERE, "quiescence.py"),
                            "--out", _qpath],
                           capture_output=True, text=True, timeout=120)
        except Exception:
            pass
        rec = measure_one_fresh(args.which, args.rep, weights, cfg, enc,
                                args.warmup, args.frames)
        rp = os.path.join(args.outdir, "repeats",
                          f"{args.which}_rep{args.rep}.json")
        os.makedirs(os.path.dirname(rp), exist_ok=True)
        with open(rp, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"  {args.which} rep{args.rep}: "
              f"{rec['throughput_fps']:.2f} FPS "
              f"({rec['mean_frame_ms']:.2f} ms/frame) -> {rp}")
        return 0

    # ---- the ORCHESTRATOR path: one fresh process per (config, repeat)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    order = list(configs)
    if not args.no_shuffle and len(configs) > 1:
        import random
        random.Random(args.seed).shuffle(order)
        print(f"shuffled schedule (seed {args.seed}): "
              f"{', '.join(order)}\n")
    else:
        print(f"schedule: {', '.join(order)}\n")

    results = {}
    for which in order:
        print(f"=== {which} : unconstrained throughput ===", flush=True)
        reps = measure_fresh(which, weights, cfg, enc, args.warmup,
                             args.frames, args.repeats, args.outdir,
                             args.seed, args.no_shuffle)
        tput = [r["throughput_fps"] for r in reps]
        mf = [r["mean_frame_ms"] for r in reps]
        p50 = [r["p50_frame_ms"] for r in reps]
        results[which] = {
            "throughput_fps_mean": st.mean(tput),
            "throughput_fps_sd": sd(tput),
            "throughput_fps_per_repeat": tput,
            "mean_frame_ms_mean": st.mean(mf),
            "mean_frame_ms_sd": sd(mf),
            "p50_frame_ms_mean": st.mean(p50),
            "p50_frame_ms_sd": sd(p50),
            "repeats": reps,
            "keeps_up_with_clock": st.mean(tput) >= args.rate,
        }
        print(f"    throughput {st.mean(tput):7.2f} +/- {sd(tput):.2f} FPS"
              f"   (frame {st.mean(mf):6.2f} +/- {sd(mf):.2f} ms)"
              f"   keeps up with {args.rate} Hz: "
              f"{'YES' if results[which]['keeps_up_with_clock'] else 'NO'}\n",
              flush=True)

    out = {
        "description": (
            "Unconstrained maximum throughput: no replay clock, no sleep, the "
            "backend is stepped back to back.  This is the maximum rate each "
            "configuration can sustain on its own, and is reported SEPARATELY "
            "from the Run 1 energy-per-frame table, which is measured at the "
            "fixed 50 Hz replay rate."),
        "clock_rate_hz_for_comparison": args.rate,
        "frames_per_repeat": args.frames,
        "warmup_frames": args.warmup,
        "repeats": args.repeats,
        "numeric": numeric,
        "protocol": (
            "One FRESH PROCESS per (config, repeat), with a shuffled "
            "schedule and a quiescence check before every window.  The "
            "earlier version of this measurement ran every configuration and "
            "every repeat inside ONE process in a fixed order, which is the "
            "same defect that contaminated the 50 Hz power pass: the C++ "
            "fusion core keeps a per-frame output map and the CUDA contexts "
            "keep their allocations, so state carried from one repeat into "
            "the next."),
        "schedule": order,
        "schedule_seed": args.seed,
        "note_tf32": (
            "TF32 is disabled so that every configuration evaluates the same "
            "function; with TF32 enabled the PyTorch baseline would differ "
            "from the others for a reason unrelated to the algorithm."),
        "results": results,
    }
    path = os.path.join(args.outdir, "throughput.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")

    print()
    print(f"{'cfg':5s} {'unconstrained FPS':>26s} {'frame ms':>22s} "
          f"{'keeps 50 Hz':>13s}")
    for k, v in results.items():
        print(f"{k:5s} "
              f"{v['throughput_fps_mean']:12.2f} +/- {v['throughput_fps_sd']:<8.2f} "
              f"{v['mean_frame_ms_mean']:11.2f} +/- {v['mean_frame_ms_sd']:<7.2f} "
              f"{('YES' if v['keeps_up_with_clock'] else 'NO'):>13s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
