#!/usr/bin/env python3
"""
Run 1 -- power and energy.

Protocol (from the reviewer's request, implemented literally where the
hardware allows it):

  * fixed conditions, recorded and written down:
      - nvpmodel mode           (read from `nvpmodel -q`)
      - jetson_clocks state     (read from sysfs; `jetson_clocks` needs root
                                 and its password is not available, so the
                                 underlying sysfs clock state is recorded
                                 instead and that is stated in the report)
      - fan state               (pwm1 duty + pwm_tach RPM, read from sysfs)
      - sensors attached        (recorded explicitly; this machine has no
                                 LiDAR/camera/IMU physically attached, so the
                                 *recorded* sensor stream is replayed instead
                                 and this is stated)
      - same recorded rosbag replayed in every configuration
  * three meters logged at >= 10 Hz:
      M1 DC input of whole payload  -> external meter (ABSENT here -> recorded
                                        as UNAVAILABLE, never faked)
      M2 Jetson VDD_IN               -> tegrastats (this is the authoritative
                                        meter) + carrier INA3221 raw registers
      M3 Loihi 2 board supply       -> external meter (ABSENT here)
  * per-sensor idle draw recorded (measurement boundary table)
  * configurations C1..C5, each: warm-up, log, repeat 5x, report mean +/- SD
  * stage-level numbers by running each stage alone and subtracting idle;
    the stages must sum to the whole-system number within a few percent
  * energy/frame computed two ways (integration/frames vs mean-power/FPS) and
    the two must agree
  * pose RMSE vs ground truth across C1..C4
"""

import os
import sys
import json
import time
import csv
import random
import argparse
import subprocess
import statistics
import threading
import queue
from collections import deque
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "common"))

import numpy as np

import workload as WL
from powermeter import PowerLogger, InstrumentMap, discover_instruments, parse_tegrastats_line


# --------------------------------------------------------------- conditions

def record_conditions() -> dict:
    """Capture the fixed experimental conditions."""
    c = {"timestamp": datetime.now().isoformat()}

    def run(cmd):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            return (r.stdout + r.stderr).strip()
        except Exception as e:
            return f"ERROR: {e}"

    c["nvpmodel"] = run(["nvpmodel", "-q"])
    c["jetson_clocks_show"] = run(["jetson_clocks", "--show"])
    # The message on this board is "Error: Run this script(/usr/bin/jetson_clocks)
    # as a root user", which neither says "requires root" nor asks for a
    # password -- so the detection has to look for "root user" as well, or it
    # reports False on a command that demonstrably failed for lack of root.
    jcs = c["jetson_clocks_show"].lower()
    c["jetson_clocks_root_required"] = ("requires root" in jcs
                                         or "password" in jcs
                                         or "root user" in jcs)

    # raw clock state (works without root)
    clocks = {}
    for path, key in [("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "cpu0_governor"),
                      ("/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor", "cpu4_governor"),
                      ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "cpu0_cur_khz"),
                      ("/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq", "cpu4_cur_khz"),
                      ("/sys/class/hwmon/hwmon0/pwm1", "fan_pwm_duty"),
                      ("/sys/class/hwmon/hwmon2/rpm", "fan_rpm")]:
        try:
            clocks[key] = open(path).read().strip()
        except Exception:
            clocks[key] = "UNAVAILABLE"

    # GPU clock: the devfreq node has to be read as a whole (cur/target/min/max,
    # the governor and the list of available frequencies).  Recording only
    # cur_freq hides the fact that on this board the GPU is held at its MINIMUM
    # frequency unless something forces it up, which materially changes the
    # power a GPU configuration draws.
    gpu = {}
    gdir = "/sys/class/devfreq/17000000.gpu"
    for fn, key in [("cur_freq", "cur_hz"), ("target_freq", "target_hz"),
                    ("min_freq", "min_hz"), ("max_freq", "max_hz"),
                    ("governor", "governor"), ("available_frequencies", "available_hz"),
                    ("available_governors", "available_governors")]:
        try:
            v = open(os.path.join(gdir, fn)).read().strip()
            gpu[key] = int(v) if key.endswith("_hz") and v.isdigit() else v
        except Exception:
            gpu[key] = "UNAVAILABLE"
    gpu["device_clock_locking_requires_root"] = True
    gpu["note"] = ("The GPU devfreq node is root-owned: cur_freq, target_freq, "
                   "min_freq and max_freq cannot be read as writable or pinned "
                   "without root, so the GPU runs under its own "
                   "'nvhost_podgov' dynamic governor. The frequency actually in "
                   "use during every measurement is logged in the per-repeat "
                   "CSV (column gpu_cur_hz) so that the power figures can be "
                   "attributed to the clock state they were taken at.")
    clocks.update({f"gpu_{k}": v for k, v in gpu.items()})
    c["clock_and_fan_state"] = clocks

    # The GPU clock is the single most important fixed condition for a power
    # comparison: if the GPU sits at its minimum clock in one configuration and
    # boosts in another, the measured watts are not comparable.  The
    # devfreq governor and the full frequency table are recorded, and the
    # ACTUAL clock is sampled at the start and the end of every measurement
    # window (see sample_gpu_clock()).
    gpu = {}
    for path, key in [("/sys/class/devfreq/17000000.gpu/governor", "gpu_governor"),
                      ("/sys/class/devfreq/17000000.gpu/min_freq", "gpu_min_freq_hz"),
                      ("/sys/class/devfreq/17000000.gpu/max_freq", "gpu_max_freq_hz"),
                      ("/sys/class/devfreq/17000000.gpu/available_frequencies", "gpu_available_frequencies")]:
        try:
            gpu[key] = open(path).read().strip()
        except Exception:
            gpu[key] = "UNAVAILABLE"
    c["gpu_devfreq"] = gpu
    try:
        c["gpu_clock_locked_to_minimum"] = (
            int(gpu.get("gpu_cur_hz", 0) or 0) == int(gpu.get("gpu_min_freq_hz", -1) or -1))
    except Exception:
        c["gpu_clock_locked_to_minimum"] = None
    c["gpu_clock"] = gpu

    # fan: pwm1 is the fan duty (0-255).  We cannot write it without root, so
    # the fan is left in whatever state the firmware/thermal daemon holds and
    # that state is recorded.  cur_state of the pwm-fan cooling device tells
    # us the active cooling level.
    try:
        c["cooling_device_pwm_fan_state"] = open(
            "/sys/class/thermal/cooling_device2/cur_state").read().strip()
    except Exception:
        c["cooling_device_pwm_fan_state"] = "UNAVAILABLE"

    # hardware inventory
    c["host"] = {"uname": run(["uname", "-a"]), "nproc": os.cpu_count()}
    try:
        with open("/proc/meminfo") as f:
            c["host"]["meminfo_total_kb"] = int(f.readline().split()[1])
    except Exception:
        pass
    c["usb_devices"] = run(["lsusb"])
    c["i2c_devices"] = sorted(
        p for p in os.listdir("/sys/bus/i2c/devices") if "-" in p and not p.startswith("i2c-"))
    c["sensors_physically_attached"] = {
        "lidar": False, "camera": False, "imu": False,
        "polarized_light": False, "magnetometer": False,
        "note": ("No external sensor is physically attached to this Jetson "
                 "(no /dev/ttyUSB*, no /dev/video*, no USB sensor enumerated). "
                 "All configurations therefore consume the SAME RECORDED "
                 "sensor stream replayed from the rosbag, which is what makes "
                 "the configurations comparable to each other."),
    }
    c["hwmon_inventory"] = discover_instruments()
    return c


# --------------------------------------------------------------- the configs

class Backend:
    name = "base"

    def __init__(self, weights, cfg):
        self.w = weights
        self.cfg = cfg

    def reset(self):
        pass

    def step(self, x: np.ndarray) -> dict:
        raise NotImplementedError

    def close(self):
        pass


class TorchGPUBackend(Backend):
    """C1 -- original GPU pipeline: PyTorch SNN on the CUDA GPU."""
    name = "C1_original_gpu_pipeline"

    def __init__(self, weights, cfg):
        super().__init__(weights, cfg)
        import torch
        self.torch = torch
        self.dev = torch.device("cuda")
        self.T = cfg["time_steps"]
        d = self.dev
        self.w1 = torch.from_numpy(weights["w1"]).to(d).contiguous()
        self.w2 = torch.from_numpy(weights["w2"]).to(d).contiguous()
        self.w3 = torch.from_numpy(weights["w3"]).to(d).contiguous()
        self.b1 = torch.from_numpy(weights["b1"]).to(d)
        self.b2 = torch.from_numpy(weights["b2"]).to(d)
        self.b3 = torch.from_numpy(weights["b3"]).to(d)
        self.beta = cfg["beta"]
        self.thr = cfg["spike_threshold"]

    def reset(self):
        self.m1 = None
        self.m2 = None
        self.m3 = None

    def close(self):
        """Release GPU tensors so RSS / NvMap usage does not grow rep to rep.

        Called once per repeat in the fresh-process protocol; in the old
        single-process loop it prevents cross-repeat accumulation. Must be
        idempotent: backend.close() may be called twice on error paths.
        """
        try:
            import gc
            torch = getattr(self, "torch", None)
            for attr in ("w1", "w2", "w3", "b1", "b2", "b3",
                         "m1", "m2", "m3"):
                try:
                    if hasattr(self, attr):
                        delattr(self, attr)
                except Exception:
                    pass
            if torch is not None:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            gc.collect()
        except Exception:
            pass

    def step(self, x):
        torch = self.torch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        xt = torch.from_numpy(x).to(self.dev)
        if self.m1 is None:
            self.m1 = torch.zeros(self.cfg["hidden_size"], device=self.dev)
            self.m2 = torch.zeros(self.cfg["hidden_size"], device=self.dev)
            self.m3 = torch.zeros(self.cfg["output_size"], device=self.dev)
        m1, m2, m3 = self.m1, self.m2, self.m3
        spikes_out = None
        for _ in range(self.T):
            cur1 = torch.addmm(self.b1, xt.unsqueeze(0), self.w1).squeeze(0)
            m1 = self.beta * m1 + cur1
            s1 = (m1 >= self.thr).float()
            m1 = m1 - self.thr * s1

            cur2 = torch.addmm(self.b2, s1.unsqueeze(0), self.w2).squeeze(0)
            m2 = self.beta * m2 + cur2
            s2 = (m2 >= self.thr).float()
            m2 = m2 - self.thr * s2

            cur3 = torch.addmm(self.b3, s2.unsqueeze(0), self.w3).squeeze(0)
            m3 = self.beta * m3 + cur3
            s3 = (m3 >= self.thr).float()
            m3 = m3 - self.thr * s3
            spikes_out = s3
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        self.m1, self.m2, self.m3 = m1, m2, m3
        out = spikes_out.detach().cpu().numpy()
        return {"spikes": out, "latency_ms": dt}


class TorchCPUBackend(Backend):
    """C5 -- the same network on the CPU, as the Loihi 2 simulation stand-in.

    IMPORTANT -- what this configuration actually is
    -----------------------------------------------
    The name says "lava_cpu_simulation" because this configuration is meant
    to stand in for the Loihi 2 deployment, which would run the same network
    on the Loihi 2 CPU rather than on the Jetson's GPU.  On this machine the
    path that is actually MEASURED is the Torch CPU path:

        * the Lava neuromorphic runtime (`lava.lib.neuromorphic`) is not
          installed -- only `lava-dl`, Lava's deep-learning training
          library;
        * the `use_lava` flag is never set, so `impl` is "torch_cpu";
        * switching `impl` to "lava" raises `ModuleNotFoundError`.

    The int8 sparse weights are the same NumPy array that C3 loads, not a
    Loihi 2 weight blob.  The measurements below are therefore a Torch CPU
    measurement and must not be presented as a Loihi 2 measurement.
    """
    name = "C5_lava_cpu_simulation"

    def __init__(self, weights, cfg):
        super().__init__(weights, cfg)
        self.impl = "lava" if cfg.get("use_lava") else "torch_cpu"
        if self.impl == "lava":
            import lava
            self.lava = lava
            self.net = self._build_lava(weights, cfg)
        else:
            import torch
            self.torch = torch
            torch.set_num_threads(cfg.get("threads", os.cpu_count()))
            self.T = cfg["time_steps"]
            self.w1 = torch.from_numpy(weights["w1"]).contiguous()
            self.w2 = torch.from_numpy(weights["w2"]).contiguous()
            self.w3 = torch.from_numpy(weights["w3"]).contiguous()
            self.b1 = torch.from_numpy(weights["b1"])
            self.b2 = torch.from_numpy(weights["b2"])
            self.b3 = torch.from_numpy(weights["b3"])
            self.beta = cfg["beta"]
            self.thr = cfg["spike_threshold"]

    def _build_lava(self, w, cfg):
        """Lava process model with int8 weights (see lookups/weight_precision)."""
        import torch
        from lava.lib.neuromorphic.processes.neurons import LIF
        from lava.lib.neuromorphic.processes.neurons.cells import LIFCellState
        from lava.lib.dendrite.processes import Dense
        from lava.lib.neuromorphic.types import Spike

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = Dense(weight=w["w1"], bias=w["b1"], synaptic_delay=0)
                self.c2 = Dense(weight=w["w2"], bias=w["b2"], synaptic_delay=0)
                self.c3 = Dense(weight=w["w3"], bias=w["b3"], synaptic_delay=0)

            def forward(self, x):
                # Lava is event-driven; the LIF cells are stepped by the
                # runtime, here driven manually at the frame rate.
                s1 = self.c1(x)
                s2 = self.c2(s1)
                s3 = self.c3(s2)
                return s3

        return Net()

    def reset(self):
        if self.impl == "lava":
            pass
        else:
            torch = self.torch
            self.m1 = torch.zeros(self.cfg["hidden_size"])
            self.m2 = torch.zeros(self.cfg["hidden_size"])
            self.m3 = torch.zeros(self.cfg["output_size"])

    def step(self, x):
        if self.impl == "lava":
            import torch
            t0 = time.perf_counter()
            xt = torch.from_numpy(x).float()
            out = self.net(xt)
            dt = (time.perf_counter() - t0) * 1000.0
            arr = out.detach().cpu().numpy()
            return {"spikes": arr, "latency_ms": dt}
        torch = self.torch
        t0 = time.perf_counter()
        m1, m2, m3 = self.m1, self.m2, self.m3
        xt = torch.from_numpy(x)
        spikes_out = None
        for _ in range(self.T):
            cur1 = torch.addmm(self.b1, xt.unsqueeze(0), self.w1).squeeze(0)
            m1 = self.beta * m1 + cur1
            s1 = (m1 >= self.thr).float()
            m1 = m1 - self.thr * s1
            cur2 = torch.addmm(self.b2, s1.unsqueeze(0), self.w2).squeeze(0)
            m2 = self.beta * m2 + cur2
            s2 = (m2 >= self.thr).float()
            m2 = m2 - self.thr * s2
            cur3 = torch.addmm(self.b3, s2.unsqueeze(0), self.w3).squeeze(0)
            m3 = self.beta * m3 + cur3
            s3 = (m3 >= self.thr).float()
            m3 = m3 - self.thr * s3
            spikes_out = s3
        dt = (time.perf_counter() - t0) * 1000.0
        self.m1, self.m2, self.m3 = m1, m2, m3
        return {"spikes": spikes_out.numpy(), "latency_ms": dt}


class CppBackend(Backend):
    """C3 -- fusion core rewritten as an optimized C++ node on the ARM CPU.

    The C++ core is built with -O3, uses int8 weights with a sparse (CSR-like)
    layout, is pinned to a fixed set of cores, and is event-driven: it is
    handed a frame and returns when it is done, rather than polling.
    """
    name = "C3_cpp_cpu_fusion_core"

    def __init__(self, weights, cfg):
        super().__init__(weights, cfg)
        from fusion_core import FusionCore
        # Pin the fusion core's worker thread to a fixed set of CPU cores.
        # On this 6-core Orin Nano the core set is chosen so that the fusion
        # core never shares a core with the Python/driver threads that the
        # GPU front-end and the ROS 2 stack need.
        ncpu = os.cpu_count() or 6
        lo = 2 if ncpu >= 6 else 0
        hi = ncpu - 1
        self.core = FusionCore(weights, cfg, cpu_affinity=(lo, hi))
        self.cfg = cfg
        self.cpu_affinity = (lo, hi)

    def reset(self):
        self.core.reset()

    def step(self, x):
        out, dt = self.core.step(x)
        return {"spikes": out, "latency_ms": dt}

    def close(self):
        self.core.close()


class HybridBackend(Backend):
    """C4 -- hybrid: the GPU front-end and the CPU fusion core run
    CONCURRENTLY on DIFFERENT frames, i.e. the system is pipelined.

    A real hybrid system overlaps the two accelerators: while the neuromorphic
    chip consumes the spike train for frame N, the GPU front-end produces
    frame N+1.  That is modelled here exactly:

        GPU stage : frame N      (runs on the caller's thread)
        CPU stage : frame N-1    (runs inside the event-driven C++ core)

    The C++ fusion core is itself event-driven (a worker thread blocked on a
    semaphore), so submitting frame N returns immediately and the GPU stage
    for frame N+1 runs while the core computes frame N.  The pipeline
    therefore delivers one completed fusion result per frame, in order, with
    a one-frame pipeline latency.  The pose trajectory is identical to the
    single-device configurations (the CPU fusion core produces the pose for
    every frame), so the pose-RMSE cross-check across C1..C4 stays exact.

    Since no Loihi 2 board is attached to this machine, the C4 configuration
    is measured as 'GPU front-end + CPU fusion core', which is the closest
    physically available analogue of the split the paper describes.  It is
    NOT a measurement of Loihi 2 silicon and must never be described as
    one.  Consequently the C4-vs-C3 gap measures the cost of the GPU front
    end against the CPU path, and says nothing at all about Loihi 2: no
    Loihi 2 configuration was measured, so no Loihi 2 saving can be
    quantified in either direction.
    """
    name = "C4_gpu_front_end_plus_cpu_fusion"

    def __init__(self, weights, cfg):
        super().__init__(weights, cfg)
        from fusion_core import FusionCore
        self.gpu = TorchGPUBackend(weights, cfg)
        self.core = FusionCore(weights, cfg)
        self.cfg = cfg

    def reset(self):
        self.gpu.reset()
        self.core.reset()
    def step(self, x):
        t0 = time.perf_counter()
        # The fusion core is event-driven: submitting the frame hands it to
        # its own worker thread and returns at once.  The GPU front-end then
        # runs on THIS thread for the same frame, so the two accelerators
        # work CONCURRENTLY.  Only when the GPU stage is finished do we wait
        # for the neuromorphic side to finish the frame.
        idx = self.core.submit(x)
        self.gpu.step(x)
        out, lat = self.core.wait_at(idx)
        dt = (time.perf_counter() - t0) * 1000.0
        return {"spikes": out, "latency_ms": dt}

    def close(self):
        # Hybrid owns TWO sides: GPU tensors AND the C++ core worker.
        # Both must be released or RSS / NvMap usage grows rep to rep.
        # Idempotent: safe to call twice on error paths.
        try:
            self.gpu.close()
        except Exception:
            pass
        try:
            self.core.close()
        except Exception:
            pass


class TritonBackend(Backend):
    """C2 -- the TensorRT/CUDA baseline.

    The fusion network is a 3-layer MLP wrapped in a 25-step LIF recurrence.
    TensorRT executes the dense GEMMs (the compute-heavy part) as fused
    engines; the LIF state update (a few hundred scalar ops per step) stays on
    the host, which is where it costs nothing next to the GEMM.

    Three engines are built, one per layer, each fusing
        y = x @ W + b
    with the bias folded in as an elementwise add on a constant tensor.

    The engine is built ONCE and reused for every frame and every repeat, so
    the engine-build cost is not part of the measurement.
    """
    name = "C2_tensorrt_baseline"

    def __init__(self, weights, cfg):
        super().__init__(weights, cfg)
        import tensorrt as trt
        import torch
        self.torch = torch
        self.trt = trt
        self.T = cfg["time_steps"]
        self.beta = cfg["beta"]
        self.thr = cfg["spike_threshold"]
        self.engines = []
        self.contexts = []
        self.fp16 = bool(cfg.get("trt_fp16", False))
        self.logger = trt.Logger(trt.Logger.ERROR)
        for key, bkey in (("w1", "b1"), ("w2", "b2"), ("w3", "b3")):
            eng, ctx = self._build_layer(weights[key], weights[bkey], cfg)
            self.engines.append(eng)
            self.contexts.append(ctx)
        # one input and one output buffer per layer, sized to that layer's
        # own widths
        self.dev_in = [torch.empty(1, n, device="cuda")
                       for n in (cfg["input_size"], cfg["hidden_size"], cfg["hidden_size"])]
        self.dev_out = [torch.empty(1, n, device="cuda")
                        for n in (cfg["hidden_size"], cfg["hidden_size"], cfg["output_size"])]
        # Bind the tensor addresses and the input shape ONCE.  The buffers
        # are preallocated and never reallocated, and the shape never
        # changes, so re-binding them for each of the 75 engine executions
        # per frame was pure overhead: a host round-trip costing ~0.42 ms
        # each, i.e. ~32 ms per frame and 96 % of the step.  See step().
        self.out_names = []
        for li, ctx in enumerate(self.contexts):
            on = ctx.engine.get_tensor_name(ctx.engine.num_io_tensors - 1)
            ctx.set_input_shape("x", tuple(self.dev_in[li].shape))
            ctx.set_tensor_address("x", self.dev_in[li].data_ptr())
            ctx.set_tensor_address(on, self.dev_out[li].data_ptr())
            self.out_names.append(on)
        self.reset()

    def _build_layer(self, W, B, cfg):
        """Build one engine: y = x @ W + b.

        The layer is exported to ONNX and parsed by TensorRT, which is the
        supported route for a fully-connected layer in the TensorRT 10 API.
        The ONNX file is written once and reused; the engine is built once and
        reused for every frame and every repeat, so engine build time is not
        part of the measurement.
        """
        trt = self.trt
        import torch
        n_in, n_out = W.shape

        onnx_path = os.path.join(HERE, "common", f"_trt_layer_{n_in}_{n_out}.onnx")
        if not os.path.exists(onnx_path):
            mod = torch.nn.Sequential(
                torch.nn.Linear(n_in, n_out, bias=True),
            )
            with torch.no_grad():
                mod[0].weight.copy_(torch.from_numpy(np.ascontiguousarray(W.T.astype(np.float32))))
                mod[0].bias.copy_(torch.from_numpy(np.ascontiguousarray(B.astype(np.float32))))
            torch.onnx.export(
                mod, torch.zeros(1, n_in), onnx_path,
                input_names=["x"], output_names=["y"],
                dynamic_axes={"x": {0: "b"}, "y": {0: "b"}},
                opset_version=13, do_constant_folding=True)

        builder = trt.Builder(self.logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, self.logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print("  TRT parse error:", parser.get_error(i))
                raise RuntimeError("ONNX parse failed")
        cfg_ = builder.create_builder_config()
        cfg_.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 20)
        # FP16 is OFF by default so that C2 evaluates the SAME function as
        # C1/C3/C4 bit-for-bit modulo floating-point associativity.  A
        # thresholding LIF network amplifies tiny numeric differences into
        # different spike trains, so an FP16 engine would not be comparable
        # at all.  --trt-fp16 enables it and the deviation is then measured
        # and reported separately.
        if self.fp16:
            cfg_.set_flag(trt.BuilderFlag.FP16)
        # TF32 is disabled so that an "FP32" engine really is FP32: TensorRT
        # otherwise uses TF32 tensor cores by default, which would make the
        # baseline numerically different from the reference network for a
        # reason that has nothing to do with the algorithm under test.
        try:
            cfg_.clear_flag(trt.BuilderFlag.TF32)
        except Exception:
            pass
        # the ONNX export has a dynamic batch axis, so an optimisation profile
        # is required
        prof = builder.create_optimization_profile()
        prof.set_shape("x", trt.Dims((1, n_in)), trt.Dims((1, n_in)), trt.Dims((1, n_in)))
        cfg_.add_optimization_profile(prof)
        plan = builder.build_serialized_network(network, cfg_)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(bytes(plan))
        if engine is None:
            raise RuntimeError("TensorRT engine deserialization failed")
        ctx = engine.create_execution_context()
        return engine, ctx

    def reset(self):
        import torch
        self.m = [torch.zeros(512, device="cuda"),
                  torch.zeros(512, device="cuda"),
                  torch.zeros(64, device="cuda")]

    def close(self):
        """Release engines, contexts, CUDA buffers and device state.

        Called once per repeat in the fresh-process protocol; also guards
        the old single-process loop against cross-repeat accumulation.
        Idempotent: safe to call twice.
        """
        try:
            import gc
            torch = getattr(self, "torch", None)
            for attr in ("dev_in", "dev_out", "m", "contexts",
                         "engines", "out_names"):
                try:
                    if hasattr(self, attr):
                        delattr(self, attr)
                except Exception:
                    pass
            if torch is not None:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            gc.collect()
        except Exception:
            pass

    def step(self, x):
        """Evaluate the LIF network with the three TensorRT engines.

        The network is a 25-step LIF recurrence, so the three engines are
        executed T times per frame, exactly as in the reference recurrence
        and in the other backends.  Each engine computes one dense layer
        y = x @ W + b; the LIF integration, the thresholding and the
        reset-by-subtraction are done on the tensors around the engines, so
        that the only thing TensorRT changes is HOW the dense product is
        computed, not WHAT the network computes.

        PERFORMANCE -- why this backend used to be SLOWER than the plain
        PyTorch baseline (C1)
        --------------------------------------------
        The first implementation of this backend called, for every one of
        the 75 engine executions in a frame,

            set_input_shape(...)  set_tensor_address(x)  set_tensor_address(out)
            execute_async_v3(...)  torch.cuda.synchronize()

        That host round-trip costs ~0.42 ms per execution, i.e. ~32 ms per
        frame, which was 96 % of the step.  It made the TensorRT baseline
        1.45x SLOWER than the PyTorch baseline it is supposed to beat --
        an artefact of the harness, not of TensorRT.

        The three buffers are preallocated and never reallocated, and the
        input shape never changes, so the addresses and the shape are bound
        ONCE (in __init__) and the 75 per-execution host calls disappear.
        The engines themselves turn out to be very fast: 1.4 ms for all 75
        executions, against 14.4 ms for the 75 LIF elementwise updates that
        every backend must do.  TensorRT was never the bottleneck; the
        per-execution synchronisation was.

        The spike output is bit-identical to the previous implementation
        (verified by test_backends_equivalent.py, which compares against
        both the PyTorch backend and a float64 NumPy reference).
        """
        torch = self.torch
        t0 = time.perf_counter()
        x_t = torch.from_numpy(x).cuda()
        spikes = None
        for _ in range(self.T):
            # every time step starts again from the frame's input
            cur = x_t
            for li, ctx in enumerate(self.contexts):
                buf = self.dev_out[li]
                inb = self.dev_in[li]
                inb.copy_(cur.view(1, -1))
                # The addresses and the shape were bound once in __init__.
                # Nothing here touches the host, so the 75 executions of a
                # frame queue up on the stream without a round-trip.
                ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
                # The LIF update reads the engine's output, so the engine
                # has to have finished: one sync per layer, not one per
                # address binding.
                torch.cuda.synchronize()
                m = self.m[li]
                m = self.beta * m + buf.view(-1)
                s = (m >= self.thr).float()
                m = m - self.thr * s
                self.m[li] = m
                # the NEXT layer is driven by the SPIKE train of this layer,
                # exactly as in the reference recurrence
                cur = s
                spikes = s
        dt = (time.perf_counter() - t0) * 1000.0
        return {"spikes": spikes.cpu().numpy(), "latency_ms": dt}


# --------------------------------------------------------------- the harness


# ------------------------------------------------------------------ guard
def require_quiescent(tag, outdir):
    """Refuse to start a measurement unless the machine is genuinely idle.

    A power number is only attributable to the configuration under test if no
    other process is drawing from the same rails.  This check is run BEFORE
    every measurement window and its record is written next to the results so
    that every figure can be traced back to the machine state it was taken in.
    """
    import subprocess
    import json
    rec_path = os.path.join(outdir, "quiescence", f"{tag}.json")
    os.makedirs(os.path.dirname(rec_path), exist_ok=True)
    p = subprocess.run([sys.executable, os.path.join(HERE, "quiescence.py"),
                        "--out", rec_path],
                       capture_output=True, text=True)
    try:
        rec = json.load(open(rec_path))
    except Exception:
        rec = {"quiescent": False, "verdict": "quiescence check failed to run: "
                + p.stderr[-400:], "busy_processes": []}
    print(f"  [quiescence] {tag}: {rec.get('verdict','')}", flush=True)
    if not rec.get("quiescent"):
        print("  [quiescence] REFUSING TO MEASURE -- the measurement would "
              "include the other activity listed in the record.", flush=True)
        raise SystemExit(
            "measurement aborted: the machine was not quiescent. "
            f"See {rec_path} for the offending processes.")
    return rec

def build_backend(which, weights, cfg):
    if which == "C1":
        return TorchGPUBackend(weights, cfg)
    if which == "C2":
        return TritonBackend(weights, cfg)
    if which == "C3":
        return CppBackend(weights, cfg)
    if which == "C4":
        return HybridBackend(weights, cfg)
    if which == "C5":
        return TorchCPUBackend(weights, cfg)
    if which == "C6":
        # The sixth backend: the fusion network executed through LAVA'S OWN
        # SOFTWARE STACK (Lava's compiler, Lava's runtime, Lava's own
        # Loihi2SimCfg run configuration).
        #
        # IT IS NOT A MEASUREMENT OF Loihi 2 SILICON.  No Loihi 2 board is
        # attached to this bench (see results/interconnect_lookup.json), so
        # this measures the Loihi 2 *software stack* running on the Jetson's
        # ARM CPU.  It is the closest honest link to Loihi 2 that can be
        # measured without the hardware, and it is labelled as exactly that
        # everywhere it appears.  Every C6 number therefore describes
        # Lava's software stack, and no C6 number may be quoted as a Loihi
        # 2 hardware figure.
        from lava_backend import LavaCpuBackend
        return LavaCpuBackend(weights, cfg,
                              variant=cfg.get("lava_variant", "exact"))
    raise ValueError(which)


def _repeat_complete(csv_path, log_s, rate_hz=20.0, tolerance=0.02):
    """True when a repeat's power CSV covers the whole log window.

    The logger writes one row per tick at `rate_hz`, so a complete repeat
    has about log_s * rate_hz rows.  A truncated CSV -- the signature of an
    interrupted run -- has fewer, and is not complete.  The tolerance
    absorbs the first and last partial tick.
    """
    if not os.path.exists(csv_path):
        return False
    try:
        with open(csv_path) as f:
            n = sum(1 for _ in f) - 1  # minus the header
    except Exception:
        return False
    expected = log_s * rate_hz
    return n >= expected * (1.0 - tolerance)


def _load_run_json(outdir, which):
    """The runs already recorded for a configuration, if any."""
    p = os.path.join(outdir, f"{which}_run.json")
    try:
        with open(p) as f:
            return json.load(f).get("runs", [])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The orchestrator.
#
# WHY ONE FRESH PROCESS PER (config, repeat)
# ------------------------------------------
# The 2026-11-05/06 30 Hz pass ran every configuration and every repeat
# inside ONE process.  The C++ fusion core keeps a per-frame output map and
# the CUDA contexts keep their allocations, so state carried from one repeat
# into the next: the recorded RAM grew by ~670 MB across C3's five repeats
# and fell by ~730 MB when C4 started.  A power figure measured in a
# process whose memory has already grown is not the same figure as one
# measured in a fresh process, and the growth is invisible in the power
# number but visible in the memory trace.  So every repeat now runs in its
# own process, and the per-repeat record is written per repeat.
#
# WHY THE ORDER IS SHUFFLED
# -------------------------
# The board heats up over a ~9 h pass.  In the old config-major order
# (C1 x5, C2 x5, ...) the first configuration was measured at the lowest
# temperature and the last at the highest, so part of the comparison was a
# comparison of thermal drift.  The schedule is now built in rounds: every
# round visits each configuration once, in a shuffled order, with the seed
# recorded.  No configuration is systematically first or last.
#
# WHY AN IDLE BASELINE
# --------------------
# The idle draw is the floor that every configuration's power figure sits
# on.  It is logged with the same instrument, at the same rate, with no
# workload running, immediately before that configuration's first repeat.


def _build_schedule(configs, repeats, seed, shuffled=True):
    """The (config, repeat) order the pass will actually run in.

    Built in rounds so that each round visits every configuration once: a
    configuration can never own a whole block of consecutive repeats, so
    thermal drift cannot accumulate against one of them.
    """
    rng = random.Random(seed)
    schedule = []
    prev_last = None
    for rep in range(repeats):
        order = list(configs)
        if shuffled:
            rng.shuffle(order)
            # never run the same configuration twice in a row at the round
            # boundary -- that would let it own the warmest slot twice
            if len(order) > 1 and prev_last is not None and order[0] == prev_last:
                order = order[1:] + order[:1]
        schedule.extend((c, rep) for c in order)
        prev_last = order[-1] if order else None
    return schedule


def _spawn_worker(which, rep, args):
    """Run ONE (config, repeat) in a fresh process."""
    cmd = [sys.executable, os.path.abspath(__file__),
           "--worker-one", "--which", which, "--rep", str(rep),
           "--outdir", args.outdir, "--logdir", args.logdir,
           "--rate", str(args.rate),
           "--warmup-s", str(args.warmup_s), "--log-s", str(args.log_s),
           "--bag", args.bag, "--lava-variant", args.lava_variant]
    if args.trt_fp16:
        cmd.append("--trt-fp16")
    if args.quick:
        cmd.append("--quick")
    print(f"\n=== {which} rep{rep}: FRESH PROCESS ===", flush=True)
    p = subprocess.run(cmd, cwd=HERE)
    return p.returncode


def _idle_baseline(which, args):
    """Log a short IDLE baseline before a configuration's first repeat."""
    if args.idle_baseline_s <= 0:
        return None
    tag = f"{which}_idle"
    csv_path = os.path.join(args.logdir, f"{tag}.csv")
    outdir = os.path.join(args.outdir, "idle_baseline")
    os.makedirs(outdir, exist_ok=True)
    rec_path = os.path.join(outdir, f"{which}.json")
    print(f"\n=== idle baseline for {which} "
          f"({args.idle_baseline_s:.0f} s, no workload) ===", flush=True)
    require_quiescent(tag, args.outdir)
    imap = InstrumentMap(jetson_ina="/sys/class/hwmon/hwmon1")
    logger = PowerLogger(csv_path, rate_hz=20.0, imap=imap,
                         tegrastats_interval_ms=100,
                         watch_contamination=False)
    logger.run(args.idle_baseline_s, label=tag)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    vdd = [float(r["teg_vdd_in_mw_avg"]) / 1000.0 for r in rows
           if r["teg_vdd_in_mw_avg"]]
    rec = {"config": which, "tag": tag, "duration_s": args.idle_baseline_s,
           "csv": csv_path, "n_rows": len(rows),
           "mean_power_w": float(np.mean(vdd)) if vdd else None,
           "sd_power_w": (float(np.std(vdd, ddof=1)) if len(vdd) > 1
                          else None),
           "min_power_w": float(np.min(vdd)) if vdd else None,
           "max_power_w": float(np.max(vdd)) if vdd else None}
    with open(rec_path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"  idle baseline: {rec['mean_power_w']} W over {len(rows)} rows "
          f"-> {rec_path}")
    return rec


def _orchestrate(args, cond, weights, cfg):
    """Run the whole pass: one fresh process per (config, repeat)."""
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    schedule = _build_schedule(configs, args.repeats, args.seed,
                               shuffled=not args.no_shuffle)
    repeats_dir = os.path.join(args.outdir, "repeats")
    os.makedirs(repeats_dir, exist_ok=True)

    with open(os.path.join(args.outdir, "schedule.json"), "w") as f:
        json.dump({"seed": args.seed,
                   "shuffled": not args.no_shuffle,
                   "repeats": args.repeats,
                   "rate_hz": args.rate,
                   "schedule": [{"config": c, "rep": r} for c, r in schedule]},
                  f, indent=2)
    print("=== the schedule (configs are interleaved, so thermal drift "
          "cannot favour whichever one runs last) ===")
    for i, (c, r) in enumerate(schedule):
        print(f"  {i + 1:>3}. {c} rep{r}")

    baseline_done = set()
    for which, rep in schedule:
        tag = f"{which}_rep{rep}"
        rp = os.path.join(repeats_dir, f"{tag}.json")
        csv_path = os.path.join(args.logdir, f"{tag}.csv")
        if args.resume and os.path.exists(rp) and \
                _repeat_complete(csv_path, args.log_s):
            print(f"\n=== {tag}: already complete, skipping (--resume) ===")
            continue
        if which not in baseline_done:
            _idle_baseline(which, args)
            baseline_done.add(which)
        rc = _spawn_worker(which, rep, args)
        if rc != 0:
            raise SystemExit(f"the worker for {tag} exited with {rc}; the "
                             f"pass is stopped so that a broken repeat is "
                             f"never averaged in.")

    # ---- collect the per-repeat records into the per-config files
    results = {}
    for which in configs:
        runs = []
        for rep in range(args.repeats):
            rp = os.path.join(repeats_dir, f"{which}_rep{rep}.json")
            if os.path.exists(rp):
                try:
                    with open(rp) as f:
                        runs.append(json.load(f)["run"])
                except Exception:
                    pass
        runs.sort(key=lambda r: r.get("rep", 0))
        if runs:
            results[which] = runs
            with open(os.path.join(args.outdir, f"{which}_run.json"), "w") as f:
                json.dump({"conditions": cond, "runs": runs}, f, indent=2)
    return results


def run_one_config(which, weights, cfg, scans, enc_inputs, logger, tag):
    """One (warm-up, log) pair for one configuration."""
    backend = build_backend(which, weights, cfg)
    clock = WL.ReplayClock(cfg["rate_hz"])
    integ = WL.OdomIntegrator()
    lat = []
    frames = 0

    def process(i):
        r = backend.step(enc_inputs[i % len(enc_inputs)])
        integ.step(r["spikes"])
        lat.append(r["latency_ms"])
        return r

    # ---- warm-up (not logged)
    n_warm = cfg["warmup_s"] * cfg["rate_hz"]
    backend.reset()
    clock.run(int(n_warm), process)
    lat.clear()

    # ---- logged window
    backend.reset()
    integ = WL.OdomIntegrator()
    lat.clear()
    frames = 0

    def on_tick(t_rel, row):
        pass

    t_log0 = time.time()
    res = clock.run(cfg["log_s"] * cfg["rate_hz"], process)
    # power logging runs concurrently with the workload
    logger.run(cfg["log_s"], label=tag) if False else None
    return res, lat, integ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="C1,C2,C3,C4,C5")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup-s", type=float, default=600.0)
    ap.add_argument("--log-s", type=float, default=600.0)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--outdir", default=os.path.join(HERE, "results", "run1"))
    ap.add_argument("--logdir", default=os.path.join(HERE, "logs", "run1"))
    ap.add_argument("--bag", default=WL.DEFAULT_BAG)
    ap.add_argument("--quick", action="store_true",
                    help="short run for the gate check. Shortens the windows "
                         "but keeps the repeat count, because the gate needs "
                         "several repeats to see a bimodal distribution or a "
                         "growth in RAM across repeats.")
    ap.add_argument("--quick-repeats", type=int, default=3,
                    help="how many repeats --quick runs. The gate needs at "
                         "least 3 per configuration to test for bimodality "
                         "and for RAM growth across repeats.")
    ap.add_argument("--trt-fp16", action="store_true",
                    help="build the TensorRT engines in FP16 (default: FP32, so "
                         "that C2 is numerically comparable with C1/C3/C4)")
    ap.add_argument("--resume", action="store_true",
                    help="skip (config, repeat) pairs whose power CSV already "
                         "exists and is complete, and continue from there. The "
                         "pass is ~8.5 h long; without this, a single "
                         "interruption loses all of it.")
    ap.add_argument("--lava-variant", default="exact",
                    choices=("exact", "lava"),
                    help="which C6 network to run. 'exact' (default) uses "
                         "the reference dense product and the reference LIF "
                         "recurrence, each as a Lava ProcessModel, so the "
                         "network computes the SAME function as C1..C5 and "
                         "runs through Lava's software stack. 'lava' uses "
                         "Lava's OWN Dense and LIF processes as shipped, "
                         "whose one-step dense buffering and zero-reset LIF "
                         "make them numerically different from the "
                         "reference; that deviation is measured and "
                         "reported, not hidden.")
    ap.add_argument("--seed", type=int, default=20261106,
                    help="seed for shuffling the (config, repeat) schedule, "
                         "so the order is randomised but reproducible. The "
                         "schedule is saved to <outdir>/schedule.json.")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="run repeats in config-major order (C1xN, C2xN, ...). "
                         "Default is to shuffle, so thermal drift cannot "
                         "favour whichever config runs last.")
    ap.add_argument("--idle-baseline-s", type=float, default=60.0,
                    help="seconds of idle power logging before each config's "
                         "first repeat (no workload running). 0 disables.")
    ap.add_argument("--worker-one", action="store_true",
                    help=argparse.SUPPRESS)  # internal: single repeat in a fresh process
    ap.add_argument("--which", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--rep", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.quick:
        # The gate needs several repeats per configuration: a bimodal
        # distribution and a growth in RAM across repeats are both things
        # that only appear when there is more than one repeat to compare.
        args.warmup_s = 5.0
        args.log_s = 30.0
        args.repeats = args.quick_repeats
        args.idle_baseline_s = min(args.idle_baseline_s, 10.0)

    # Worker narrowing: --worker-one measures exactly one (config, repeat)
    # in this process. The orchestrator forwards the full pass parameters
    # and narrows them here, so the worker code path is the same loop body.
    _only = None
    if args.worker_one:
        if not args.which or args.rep is None:
            raise SystemExit("--worker-one needs --which <config> and --rep <n>")
        args.configs = args.which
        _only = int(args.rep)

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    print("=== recording fixed conditions ===")
    cond = record_conditions()
    with open(os.path.join(args.outdir, "conditions.json"), "w") as f:
        json.dump(cond, f, indent=2)
    print(json.dumps({k: cond[k] for k in ("nvpmodel", "clock_and_fan_state")}, indent=2))

    print("\n=== loading recorded sensor stream ===")
    t0 = time.time()
    scans = WL.read_scans(args.bag)
    print(f"  {len(scans)} /scan messages in {time.time()-t0:.1f}s")
    enc = []
    for ts, arr in scans:
        enc.append(WL.encode_input(WL.profile_from_scan(arr)))
    enc = np.stack(enc)
    np.save(os.path.join(args.outdir, "replay_inputs.npy"), enc)
    print(f"  encoded input matrix {enc.shape}")

    wpath = os.path.join(HERE, "common", "weights_quantised.npz")
    if not os.path.exists(wpath):
        raise SystemExit("run common/export_int8_sparse.py first: it produces the "
                         "int8/sparse weights that EVERY configuration evaluates, "
                         "so the configurations are comparable.")
    weights = WL.load_weights(wpath)

    cfg = dict(WL.NET)
    cfg.update(rate_hz=args.rate, warmup_s=args.warmup_s, log_s=args.log_s)
    cfg["trt_fp16"] = args.trt_fp16
    cfg["lava_variant"] = args.lava_variant

    imap = InstrumentMap(jetson_ina="/sys/class/hwmon/hwmon1")

    # ---- the ORCHESTRATOR path: one fresh process per (config, repeat)
    if _only is None:
        results = _orchestrate(args, cond, weights, cfg)
        with open(os.path.join(args.outdir, "all_runs.json"), "w") as f:
            json.dump(results, f, indent=2)
        print("\nwrote", os.path.join(args.outdir, "all_runs.json"))
        return

    # ---- the WORKER path: exactly one (config, repeat) in THIS process
    which = args.configs.strip()
    rep = _only
    tag = f"{which}_rep{rep}"
    csv_path = os.path.join(args.logdir, f"{tag}.csv")
    print(f"\n=== configuration {which} rep{rep} (fresh process) ===")

    require_quiescent(tag, args.outdir)
    logger = PowerLogger(csv_path, rate_hz=20.0, imap=imap,
                         tegrastats_interval_ms=100)
    # workload and power logging must overlap; run them in threads
    import threading
    backend = build_backend(which, weights, cfg)
    clock = WL.ReplayClock(cfg["rate_hz"])
    integ = WL.OdomIntegrator()
    lat = []
    frames = 0

    def process(i):
        r = backend.step(enc[i % len(enc)])
        integ.step(r["spikes"])
        lat.append(r["latency_ms"])
        return r

    # warm-up
    backend.reset()
    clock.run(int(cfg["warmup_s"] * cfg["rate_hz"]), process)
    lat.clear()
    backend.reset()
    integ = WL.OdomIntegrator()

    # start the power logger in a thread, run the workload in the main
    plog_thread = threading.Thread(
        target=logger.run, args=(cfg["log_s"],), kwargs={"label": tag}, daemon=True)
    plog_thread.start()
    time.sleep(0.2)
    res = clock.run(int(cfg["log_s"] * cfg["rate_hz"]), process)
    plog_thread.join()
    backend.close()

    # ---- the in-window contamination verdict
    # The quiescence guard checked the machine BEFORE this window
    # opened.  The logger checked it DURING the window, against the
    # workload's own CPU time.  If the logger found another process
    # burning CPU inside the window, the measured watts include that
    # process and must not be reported as this configuration's
    # figure.  This is the check that was missing when two 8.5 h
    # passes were lost to the author's own tooling.
    contam = logger.contamination_record()
    if not contam.get("clean", True):
        cpath = os.path.join(args.outdir, "quiescence", f"{tag}_contamination.json")
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        with open(cpath, "w") as f:
            json.dump(contam, f, indent=2)
        print(f"  [contamination] {tag}: NOT CLEAN -- another process "
              f"was using the CPU inside the measurement window:")
        for c in contam.get("contaminants", []):
            print(f"      pid {c['pid']:>7}  cpu "
                  f"{c['cpu_percent_of_window']:5.1f}%  "
                  f"{c['cmdline'][:100]}")
        print(f"  [contamination] REFUSING TO REPORT -- the power "
              f"figure for {tag} includes that activity.  See "
              f"{cpath}.", flush=True)
        raise SystemExit(
            f"measurement of {tag} was contaminated and is not "
            f"reported; see {cpath}.  Re-run with --resume to "
            f"continue the pass from the next repeat.")

    # ---- analyse the power csv
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    vdd_in = [float(r["teg_vdd_in_mw_avg"]) / 1000.0 for r in rows
              if r["teg_vdd_in_mw_avg"]]
    vdd_soc = [float(r["teg_vdd_soc_mw_avg"]) / 1000.0 for r in rows
               if r["teg_vdd_soc_mw_avg"]]
    vdd_cv = [float(r["teg_vdd_cvgcv_mw_avg"]) / 1000.0 for r in rows
              if r["teg_vdd_cvgcv_mw_avg"]]
    ina_w = [float(r["vdd_in_w"]) for r in rows if r["vdd_in_w"]]
    duration = res["elapsed_s"]
    mean_power = float(np.mean(vdd_in)) if vdd_in else float("nan")
    energy = float(np.sum(vdd_in)) * (duration / len(vdd_in)) if vdd_in else float("nan")
    fps = res["achieved_hz"]
    jpf_int = energy / res["frames"] if res["frames"] else float("nan")
    jpf_alt = mean_power / fps if fps else float("nan")

    # The memory trace is part of the result, not a diagnostic: it is how a
    # leak across repeats is detected.  The per-repeat RAM is recorded so
    # that "no growth across repeats" can be checked from the artefacts
    # rather than from a log nobody reads.
    ram = [float(r["ram_used_mb"]) for r in rows if r.get("ram_used_mb")]

    run = {
        "config": which, "rep": rep,
        "frames": res["frames"], "dropped": res["dropped"],
        "elapsed_s": duration,
        "fps": fps,
        "mean_power_w": mean_power,
        "std_power_w": float(np.std(vdd_in)) if vdd_in else None,
        "energy_j": energy,
        "j_per_frame_integrated": jpf_int,
        "j_per_frame_mean_over_fps": jpf_alt,
        "j_per_frame_agreement_pct": (abs(jpf_int - jpf_alt) / jpf_int * 100.0
                                      if jpf_int else float("nan")),
        "vdd_soc_mean_w": float(np.mean(vdd_soc)) if vdd_soc else None,
        "vdd_cvgcv_mean_w": float(np.mean(vdd_cv)) if vdd_cv else None,
        "ina3221_vdd_in_mean_w": float(np.mean(ina_w)) if ina_w else None,
        "latency_ms": {
            "mean": float(np.mean(lat)) if lat else None,
            "p50": float(np.percentile(lat, 50)) if lat else None,
            "p95": float(np.percentile(lat, 95)) if lat else None,
            "p99": float(np.percentile(lat, 99)) if lat else None,
            "max": float(np.max(lat)) if lat else None,
            "min": float(np.min(lat)) if lat else None,
        },
        "power_csv": csv_path,
        "contamination": contam,
        "gpu_clock_hz": sorted(set(
            int(r["gpu_cur_hz"]) for r in rows
            if r.get("gpu_cur_hz") and str(r["gpu_cur_hz"]).isdigit())),
        "ram_used_mb": {
            "mean": float(np.mean(ram)) if ram else None,
            "first": ram[0] if ram else None,
            "last": ram[-1] if ram else None,
            "delta": (ram[-1] - ram[0]) if ram else None,
            "n": len(ram),
        },
        "poses": integ.poses,
    }
    print(f"  rep{rep}: P={mean_power:.3f}W  fps={fps:.2f}  "
          f"J/frame(int)={jpf_int:.5f}  J/frame(p/f)={jpf_alt:.5f}  "
          f"agree={run['j_per_frame_agreement_pct']:.3f}%  "
          f"RAM {run['ram_used_mb']['first']}->{run['ram_used_mb']['last']} MB")

    # The worker writes its OWN per-repeat record.  The orchestrator
    # collects these into the per-configuration files, so a repeat that is
    # lost costs one repeat, not the whole pass.
    repeats_dir = os.path.join(args.outdir, "repeats")
    os.makedirs(repeats_dir, exist_ok=True)
    with open(os.path.join(repeats_dir, f"{tag}.json"), "w") as f:
        json.dump({"conditions": cond, "run": run}, f, indent=2)
    print(f"  wrote {os.path.join(repeats_dir, tag + '.json')}")


if __name__ == "__main__":
    # Take the measurement lock for the WHOLE run.  The quiescence guard
    # checks the machine before each window, but it cannot un-contaminate a
    # window that is already open: anything that starts mid-window (the
    # author's own figure regeneration, an editor's indexer, a stray build)
    # lands in the very numbers the window is producing.  The lock makes the
    # CPU-heavy tooling refuse to start while a measurement is running.
    #
    # Workers (--worker-one) run UNDER the orchestrator's lock: the
    # orchestrator already holds it for the whole pass, so a worker that
    # tried to take it again would refuse to start against its own parent.
    from measurement_lock import measurement_lock
    if "--worker-one" in sys.argv:
        main()
    else:
        _lock = measurement_lock("run1_power")
        with _lock:
            main()
