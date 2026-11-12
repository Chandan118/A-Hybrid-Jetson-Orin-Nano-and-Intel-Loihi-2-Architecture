#!/usr/bin/env python3
"""
Run 1c -- output-equivalence check (pose RMSE).

The reviewer asks: "Check that C1-C4 produce the same output, using pose
RMSE against ground truth."

What ground truth exists on this machine
----------------------------------------
The recorded bag (rosbag_exp4_20260404_021816) contains /scan (2746 messages,
the sensor stream used by every configuration), /odom (7268 messages, but
EVERY pose in it is exactly (0,0,0) -- the vehicle was stationary and the
publisher emitted the identity pose), /amcl_pose (1 message) and /plan (0
messages).  There is therefore NO usable ground-truth trajectory in the
recording.

The reference used here is consequently the DEFINITIVE one: the same LIF
recurrence evaluated in float64 NumPy, which is the mathematical definition
of the network that every backend is supposed to implement.  A backend that
implements the network correctly must reproduce that trajectory.

For each configuration we therefore report
  * RMSE(x, y) of the produced trajectory against the float64 reference
    trajectory, over the identical frame sequence;
  * the pairwise RMSE between the configurations, which is the direct
    "C1-C4 produce the same output" test.
"""

import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
sys.path.insert(0, HERE)

import numpy as np

import workload as WL
from run1_power import build_backend


def reference_trajectory(enc, cfg, n_frames):
    """float64 NumPy evaluation of the LIF network -> pose trajectory.

    This is the reference implementation: 25 LIF time steps per frame,
    beta = 0.9, threshold = 0.5, reset-by-subtraction, exactly as in the
    backends, but in float64 so that no backend's arithmetic can bias it.
    """
    beta = cfg["beta"]
    thr = cfg["spike_threshold"]
    T = cfg["time_steps"]
    w1, w2, w3 = (cfg["_w"][k] for k in ("w1", "w2", "w3"))
    b1, b2, b3 = (cfg["_w"][k] for k in ("b1", "b2", "b3"))

    m1 = np.zeros(cfg["hidden_size"], dtype=np.float64)
    m2 = np.zeros(cfg["hidden_size"], dtype=np.float64)
    m3 = np.zeros(cfg["output_size"], dtype=np.float64)

    integ = WL.OdomIntegrator()
    for i in range(n_frames):
        x = enc[i % len(enc)].astype(np.float64)
        spikes = None
        for _ in range(T):
            m1 = beta * m1 + (x @ w1 + b1)
            s1 = (m1 >= thr).astype(np.float64)
            m1 = m1 - thr * s1
            m2 = beta * m2 + (s1 @ w2 + b2)
            s2 = (m2 >= thr).astype(np.float64)
            m2 = m2 - thr * s2
            m3 = beta * m3 + (s2 @ w3 + b3)
            spikes = (m3 >= thr).astype(np.float64)
            m3 = m3 - thr * spikes
        integ.step(spikes)
    return np.asarray(integ.poses, dtype=np.float64)


def pose_rmse(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return float("nan"), 0
    a = np.asarray(a[:n], dtype=np.float64)
    b = np.asarray(b[:n], dtype=np.float64)
    d = np.sqrt((a[:, 0] - b[:, 0]) ** 2 + (a[:, 1] - b[:, 1]) ** 2)
    return float(np.sqrt(np.mean(d ** 2))), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="C1,C2,C3,C4,C5")
    ap.add_argument("--frames", type=int, default=0,
                    help="frames to compare (0 = all recorded frames)")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "run1",
                                                  "pose_equivalence.json"))
    ap.add_argument("--bag", default=WL.DEFAULT_BAG)
    args = ap.parse_args()

    scans = WL.read_scans(args.bag)
    enc = np.stack([WL.encode_input(WL.profile_from_scan(a)) for _, a in scans])
    n_frames = args.frames or len(enc)
    print(f"reference run: {n_frames} frames from the recorded bag")

    cfg = dict(WL.NET)
    cfg["rate_hz"] = args.rate
    weights = WL.load_weights(os.path.join(HERE, "common", "weights_quantised.npz"))
    cfg["_w"] = {k: weights[k].astype(np.float64) for k in weights}

    # The GPU's floating-point behaviour is part of the result: PyTorch's
    # cuBLAS path may use TF32 tensor cores by default, which would make the
    # PyTorch baseline differ from the float64 reference for a reason that has
    # nothing to do with the algorithm under test.  The setting is therefore
    # recorded, and TF32 is switched OFF so that every configuration
    # evaluates the same function.
    numeric = {}
    try:
        import torch
        numeric["torch_version"] = torch.__version__
        numeric["cudnn_allow_tf32_before"] = bool(torch.backends.cudnn.allow_tf32)
        numeric["matmul_allow_tf32_before"] = bool(
            torch.backends.cuda.matmul.allow_tf32)
        numeric["float32_matmul_precision_before"] = str(
            torch.get_float32_matmul_precision())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
        numeric["matmul_allow_tf32_after"] = bool(
            torch.backends.cuda.matmul.allow_tf32)
        numeric["float32_matmul_precision_after"] = str(
            torch.get_float32_matmul_precision())
        print("numeric settings:", numeric)
    except Exception as e:
        numeric["error"] = str(e)

    t0 = __import__("time").time()
    ref = reference_trajectory(enc, cfg, n_frames)
    print(f"float64 reference trajectory: {len(ref)} poses, "
          f"final=({ref[-1,0]:.6f}, {ref[-1,1]:.6f}, {ref[-1,2]:.6f}), "
          f"{__import__('time').time()-t0:.1f}s")

    results = {}
    poses_by_cfg = {}
    for which in [c.strip() for c in args.configs.split(",") if c.strip()]:
        b = build_backend(which, weights, cfg)
        b.reset()
        integ = WL.OdomIntegrator()
        for i in range(n_frames):
            r = b.step(enc[i % len(enc)])
            integ.step(r["spikes"])
        b.close()
        p = np.asarray(integ.poses, dtype=np.float64)
        rm, n = pose_rmse(p, ref)
        poses_by_cfg[which] = p
        results[which] = {"rmse_vs_float64_reference_m": rm, "n_poses": n,
                           "final_pose": p[-1].tolist()}
        print(f"  {which:<4} RMSE vs float64 reference = {rm:.9f} m  "
              f"final=({p[-1,0]:.6f}, {p[-1,1]:.6f})")

    # pairwise
    keys = list(results)
    pairwise = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            rm, _ = pose_rmse(poses_by_cfg[a], poses_by_cfg[b])
            pairwise[f"{a}_vs_{b}"] = rm

    out = {
        "numeric_settings": numeric,
        "frames_compared": n_frames,
        "reference": "float64 NumPy evaluation of the same LIF recurrence "
                     "(the mathematical definition of the network)",
        "ground_truth_in_recording": (
            "NONE: /odom in the recorded bag contains only (0,0,0) poses, "
            "/amcl_pose has 1 message and /plan has 0.  The float64 "
            "reference is therefore used instead of a recorded trajectory."),
        "vs_reference": results,
        "pairwise_rmse_m": pairwise,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n--- pairwise pose RMSE (m) ---")
    for k, v in pairwise.items():
        print(f"  {k:<18} {v:.9f}")
    print("\nwrote", args.out)



if __name__ == "__main__":
    # Hold the measurement lock for the whole run: the quiescence guard can
    # only check the machine BEFORE a window opens, so anything that starts
    # mid-window contaminates the numbers being produced right now.  See
    # measurement_lock.py.
    from measurement_lock import measurement_lock
    with measurement_lock("run1c_pose_check"):
        main()
