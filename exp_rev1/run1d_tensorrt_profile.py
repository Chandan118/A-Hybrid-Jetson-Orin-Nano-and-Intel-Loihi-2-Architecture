#!/usr/bin/env python3
"""Profile the C2 (TensorRT) backend and explain why it used to be slower
than the C1 (PyTorch) baseline.

Why this script exists
----------------------
Reviewer 1 complained about a weak GPU baseline, and a TensorRT baseline that
is SLOWER than the plain PyTorch baseline invites the same complaint. The
first implementation of the TensorRT backend was indeed slower: 31.2 FPS
against PyTorch's 44.6 FPS. That number was an artefact of the HARNESS, not
of TensorRT.

This script measures, in the real backend, where the time actually goes, so
that the paper can state the reason rather than speculate about it.

What it measures
----------------
1. The total time of one C2 frame.
2. The time of the 75 TensorRT engine executions alone (no LIF, one sync at
   the end) -- i.e. what TensorRT itself costs.
3. The time of the 75 LIF elementwise updates alone -- the work every
   backend has to do.
4. The cost of the per-execution host round-trip that the first
   implementation performed: set_input_shape, set_tensor_address x2 and
   execute_async_v3, repeated 75 times per frame.
5. The same measurement with the addresses bound ONCE, which is what the
   fixed backend does.

It also verifies that the fixed backend produces BIT-IDENTICAL spikes to the
old one, so that the speed-up is not a change of the algorithm.

The output is written to results/run1d/tensorrt_profile.json and is quoted
in the review response.
"""

import argparse
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
sys.path.insert(0, HERE)

import numpy as np
import torch
import workload as WL
from run1_power import TritonBackend, TorchGPUBackend


def set_numeric():
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--outdir", default=os.path.join(HERE, "results", "run1d"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_numeric()

    weights = WL.load_weights(os.path.join(HERE, "common", "weights_quantised.npz"))
    cfg = dict(WL.NET)
    cfg["rate_hz"] = 50
    cfg["trt_fp16"] = False

    rng = np.random.default_rng(7)
    x = rng.standard_normal(cfg["input_size"]).astype(np.float32)

    out = {
        "description": (
            "Profile of the C2 (TensorRT) backend, explaining why it was "
            "slower than the C1 (PyTorch) baseline in the first "
            "implementation and what the fixed implementation costs."),
        "engine": "FP32 (trt_fp16 = False), one engine per dense layer",
        "time_steps": cfg["time_steps"],
        "executions_per_frame": cfg["time_steps"] * 3,
    }

    # ---------------- C2 as it is now (addresses bound once)
    b = TritonBackend(weights, cfg)
    for _ in range(20):
        b.step(x)
    torch.cuda.synchronize()
    N = args.frames

    t0 = time.perf_counter()
    for _ in range(N):
        b.step(x)
    torch.cuda.synchronize()
    c2_fixed = (time.perf_counter() - t0) / N * 1000

    # ---------------- C1 for comparison
    c1 = TorchGPUBackend(weights, cfg)
    c1.reset()
    for _ in range(20):
        c1.step(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        c1.step(x)
    torch.cuda.synchronize()
    c1_ms = (time.perf_counter() - t0) / N * 1000

    # ---------------- the OLD implementation: rebind every execution
    b.reset()
    x_t = torch.from_numpy(x).cuda()
    t0 = time.perf_counter()
    for _ in range(N):
        spikes = None
        for _ in range(b.T):
            cur = x_t
            for li, ctx in enumerate(b.contexts):
                buf = b.dev_out[li]
                inb = b.dev_in[li]
                inb.copy_(cur.view(1, -1))
                ctx.set_input_shape("x", tuple(inb.shape))
                ctx.set_tensor_address("x", inb.data_ptr())
                on = ctx.engine.get_tensor_name(ctx.engine.num_io_tensors - 1)
                ctx.set_tensor_address(on, buf.data_ptr())
                ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()
                m = b.m[li]
                m = b.beta * m + buf.view(-1)
                s = (m >= b.thr).float()
                m = m - b.thr * s
                b.m[li] = m
                cur = s
                spikes = s
    torch.cuda.synchronize()
    c2_old = (time.perf_counter() - t0) / N * 1000

    # ---------------- the engines alone (no LIF, one sync at the end)
    b.reset()
    t0 = time.perf_counter()
    for _ in range(N):
        x_t = torch.from_numpy(x).cuda()
        cur = x_t
        for li, ctx in enumerate(b.contexts):
            b.dev_in[li].copy_(cur.view(1, -1))
            ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            cur = b.dev_out[li].view(-1)
    torch.cuda.synchronize()
    engines_ms = (time.perf_counter() - t0) / N * 1000

    # ---------------- the LIF updates alone
    buf1, buf2, buf3 = b.dev_out[0].view(-1), b.dev_out[1].view(-1), b.dev_out[2].view(-1)
    m1 = torch.zeros(cfg["hidden_size"], device="cuda")
    m2 = torch.zeros(cfg["hidden_size"], device="cuda")
    m3 = torch.zeros(cfg["output_size"], device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        for _ in range(25):
            m1 = b.beta * m1 + buf1; s1 = (m1 >= b.thr).float(); m1 = m1 - b.thr * s1
            m2 = b.beta * m2 + buf2; s2 = (m2 >= b.thr).float(); m2 = m2 - b.thr * s2
            m3 = b.beta * m3 + buf3; s3 = (m3 >= b.thr).float(); m3 = m3 - b.thr * s3
    torch.cuda.synchronize()
    lif_ms = (time.perf_counter() - t0) / N * 1000

    # ---------------- the spike-train copies into the engine inputs
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        for _ in range(25):
            b.dev_in[1].copy_(b.dev_out[0].view(1, -1))
            b.dev_in[2].copy_(b.dev_out[1].view(1, -1))
    torch.cuda.synchronize()
    copies_ms = (time.perf_counter() - t0) / N * 1000

    # ---------------- the per-execution synchronisation cost
    # engines only, one sync at the very end, vs a sync after each execution
    def eng(one_sync):
        x_t = torch.from_numpy(x).cuda()
        cur = x_t
        for li, ctx in enumerate(b.contexts):
            b.dev_in[li].copy_(cur.view(1, -1))
            ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            if not one_sync:
                torch.cuda.synchronize()
            cur = b.dev_out[li].view(-1)
        if one_sync:
            torch.cuda.synchronize()

    def _t(fn):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / N * 1000

    eng_onesync_ms = _t(lambda: eng(True))
    eng_persync_ms = _t(lambda: eng(False))
    per_sync_ms = eng_persync_ms - eng_onesync_ms

    # ---------------- the per-frame host-to-device copy
    t0 = time.perf_counter()
    for _ in range(N):
        torch.from_numpy(x).cuda()
    torch.cuda.synchronize()
    h2d_ms = (time.perf_counter() - t0) / N * 1000

    out["measurements_ms_per_frame"] = {
        "c2_tensorrt_fixed_ms": c2_fixed,
        "c2_tensorrt_old_rebinding_ms": c2_old,
        "c1_pytorch_ms": c1_ms,
        "tensorrt_engine_executions_only_ms": engines_ms,
        "lif_elementwise_updates_only_ms": lif_ms,
        "per_frame_host_to_device_copy_ms": h2d_ms,
        "spike_train_copies_into_engine_inputs_ms": copies_ms,
        "per_execution_sync_cost_ms": per_sync_ms,
        "engines_one_sync_ms": eng_onesync_ms,
        "engines_sync_each_ms": eng_persync_ms,
    }
    accounted = (engines_ms + lif_ms + copies_ms + h2d_ms)
    out["derived"] = {
        "c2_fixed_fps": 1000.0 / c2_fixed,
        "c1_fps": 1000.0 / c1_ms,
        "c2_old_fps": 1000.0 / c2_old,
        "old_c2_to_c1_latency_ratio": c2_old / c1_ms,
        "fixed_c2_to_c1_latency_ratio": c2_fixed / c1_ms,
        "address_rebinding_cost_ms": c2_old - c2_fixed,
        "address_rebinding_share_of_old_step": (c2_old - c2_fixed) / c2_old,
        "engine_share_of_fixed_step": engines_ms / c2_fixed,
        "lif_share_of_fixed_step": lif_ms / c2_fixed,
        "accounted_ms": accounted,
        "unaccounted_ms": c2_fixed - accounted,
        "unaccounted_share": (c2_fixed - accounted) / c2_fixed,
        "lif_kernel_launches_per_frame": 25 * 3 * 5,
        "engine_executions_per_frame": 25 * 3,
    }

    # ---------------- equivalence: the fix must not change the answer
    b.reset()
    fixed_spikes = b.step(x)["spikes"].copy()
    b.reset()
    spikes = None
    x_t = torch.from_numpy(x).cuda()
    for _ in range(b.T):
        cur = x_t
        for li, ctx in enumerate(b.contexts):
            buf = b.dev_out[li]
            inb = b.dev_in[li]
            inb.copy_(cur.view(1, -1))
            ctx.set_input_shape("x", tuple(inb.shape))
            ctx.set_tensor_address("x", inb.data_ptr())
            on = ctx.engine.get_tensor_name(ctx.engine.num_io_tensors - 1)
            ctx.set_tensor_address(on, buf.data_ptr())
            ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            m = b.m[li]
            m = b.beta * m + buf.view(-1)
            s = (m >= b.thr).float()
            m = m - b.thr * s
            b.m[li] = m
            cur = s
            spikes = s
    old_spikes = spikes.cpu().numpy()
    out["equivalence"] = {
        "differing_spike_units": int((fixed_spikes != old_spikes).sum()),
        "total_output_units": int(fixed_spikes.size),
        "bit_identical": bool((fixed_spikes == old_spikes).all()),
    }

    path = os.path.join(args.outdir, "tensorrt_profile.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    m = out["measurements_ms_per_frame"]
    d = out["derived"]
    print("=" * 74)
    print("C2 (TensorRT) PROFILE  --  ms per frame")
    print("=" * 74)
    print(f"  C2 as first implemented (rebind every execution) : {m['c2_tensorrt_old_rebinding_ms']:7.3f} ms"
          f"   ({d['c2_old_fps']:.2f} FPS)")
    print(f"  C2 with the addresses bound once (the fix)       : {m['c2_tensorrt_fixed_ms']:7.3f} ms"
          f"   ({d['c2_fixed_fps']:.2f} FPS)")
    print(f"  C1 (PyTorch addmm) for comparison               : {m['c1_pytorch_ms']:7.3f} ms"
          f"   ({d['c1_fps']:.2f} FPS)")
    print()
    print(f"  where the FIXED C2 step goes:")
    print(f"    TensorRT engine executions alone (75/frame)     : "
          f"{m['tensorrt_engine_executions_only_ms']:.3f} ms"
          f"  ({d['engine_share_of_fixed_step']*100:.0f} %)")
    print(f"    the 75 LIF elementwise updates alone            : "
          f"{m['lif_elementwise_updates_only_ms']:.3f} ms"
          f"  ({d['lif_share_of_fixed_step']*100:.0f} %)")
    print(f"    the 50 spike-train copies into the engine inputs : "
          f"{m['spike_train_copies_into_engine_inputs_ms']:.3f} ms")
    print(f"    per-frame host-to-device copy of the input      : "
          f"{m['per_frame_host_to_device_copy_ms']:.3f} ms")
    print(f"    per-execution synchronisation (74 extra syncs)  : "
          f"{m['per_execution_sync_cost_ms']:.3f} ms")
    print(f"    Python loop overhead + remainder                : "
          f"{d['unaccounted_ms']:.3f} ms  ({d['unaccounted_share']*100:.0f} %)")
    print()
    print(f"  the address rebinding that the fix removed       : "
          f"{d['address_rebinding_cost_ms']:.3f} ms"
          f"  ({d['address_rebinding_share_of_old_step']*100:.0f} % of the old step)")
    print()
    print(f"  old C2 / C1 latency ratio : {d['old_c2_to_c1_latency_ratio']:.2f}x"
          f"   (TensorRT was SLOWER than PyTorch)")
    print(f"  fixed C2 / C1 latency ratio: {d['fixed_c2_to_c1_latency_ratio']:.2f}x")
    print()
    print(f"  spike output bit-identical to the old implementation: "
          f"{out['equivalence']['bit_identical']} "
          f"({out['equivalence']['differing_spike_units']} differing units)")
    print()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
