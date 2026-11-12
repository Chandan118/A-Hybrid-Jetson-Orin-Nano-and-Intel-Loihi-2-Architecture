#!/usr/bin/env python3
"""
ctypes wrapper for the C++ fusion core (C3).

The core is event-driven: `step()` posts an event and blocks until the worker
has finished the frame, so the latency returned is the pure compute latency
with no queueing.
"""

import ctypes
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(HERE, "libfusion_core.so")
WEIGHTS_BIN = os.path.join(HERE, "fusion_weights_int8_sparse.bin")


def build(extra_flags=("-O3", "-march=armv8.2-a", "-ffast-math")):
    import subprocess
    cmd = ["g++", "-shared", "-fPIC", "-std=c++17",
           "-I", HERE, *extra_flags,
           os.path.join(HERE, "fusion_core.cpp"),
           "-o", LIB_PATH, "-lpthread"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("build failed:\n" + r.stderr)
    return LIB_PATH


class FusionCore:
    def __init__(self, weights=None, cfg=None, cpu_affinity=None, inputs=None):
        if not os.path.exists(LIB_PATH):
            build()
        self.lib = ctypes.CDLL(LIB_PATH)
        L = self.lib
        L.fc_create.argtypes = [ctypes.c_char_p]
        L.fc_create.restype = ctypes.c_void_p
        L.fc_configure.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                   ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int]
        L.fc_configure.restype = ctypes.c_int
        L.fc_set_affinity.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        L.fc_set_affinity.restype = ctypes.c_int
        L.fc_reset.argtypes = [ctypes.c_void_p]
        L.fc_set_input.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_float)]
        L.fc_set_input.restype = ctypes.c_int
        L.fc_submit.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        L.fc_submit.restype = ctypes.c_int
        L.fc_wait.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_double)]
        L.fc_wait.restype = ctypes.c_int
        L.fc_last_output.argtypes = [ctypes.c_void_p]
        L.fc_last_output.restype = ctypes.POINTER(ctypes.c_float)
        L.fc_last_output_at.argtypes = [ctypes.c_void_p, ctypes.c_int64,
                                       ctypes.POINTER(ctypes.c_float)]
        L.fc_last_output_at.restype = ctypes.c_int
        L.fc_latency_at.argtypes = [ctypes.c_void_p, ctypes.c_int64,
                                    ctypes.POINTER(ctypes.c_double)]
        L.fc_latency_at.restype = ctypes.c_int
        L.fc_last_pose.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        L.fc_frames_done.argtypes = [ctypes.c_void_p]
        L.fc_frames_done.restype = ctypes.c_int64
        L.fc_destroy.argtypes = [ctypes.c_void_p]
        L.fc_layer_nnz.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64),
                                   ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]

        self.h = L.fc_create(WEIGHTS_BIN.encode())
        if not self.h:
            raise RuntimeError("fc_create failed")
        self.cfg = cfg or {}
        self._configure(inputs, cpu_affinity)
        self.frame = 0

    def _configure(self, inputs, cpu_affinity):
        arr = None
        if inputs is not None:
            arr = np.ascontiguousarray(inputs, dtype=np.float32)
        rc = self.lib.fc_configure(self.h, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                                   if arr is not None else None,
                                   arr.shape[0] if arr is not None else 0,
                                   arr.shape[1] if arr is not None else 0,
                                   int(self.cfg.get("rate_hz", 50)),
                                   int(cpu_affinity[0]) if cpu_affinity else 0,
                                   int(cpu_affinity[1]) if cpu_affinity else 0)
        if rc != 0:
            raise RuntimeError(f"fc_configure failed rc={rc}")
        if cpu_affinity:
            self.lib.fc_set_affinity(self.h, int(cpu_affinity[0]), int(cpu_affinity[1]))

    def set_inputs(self, inputs):
        self._configure(inputs, None)

    def reset(self):
        self.lib.fc_reset(self.h)
        # NOTE: the frame counter is deliberately NOT reset.  The C++ core's
        # `done` counter is monotonic (it counts completed frames since the
        # core was created), so resetting the frame index here would make
        # fc_wait() return immediately for the wrong frame.  fc_reset() clears
        # the per-frame outputs and the LIF state, which is what a new
        # measurement window needs.
        return None

    def step(self, x=None):
        lat = ctypes.c_double(0.0)
        if x is not None:
            xa = np.ascontiguousarray(x, dtype=np.float32)
            self.lib.fc_set_input(self.h, self.frame,
                                  xa.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        self.lib.fc_submit(self.h, self.frame)
        rc = self.lib.fc_wait(self.h, self.frame, ctypes.byref(lat))
        if rc != 0:
            raise RuntimeError("fc_wait failed")
        out = np.ctypeslib.as_array(self.lib.fc_last_output(self.h), shape=(64,)).copy()
        self.frame += 1
        return out, lat.value

    # ---- asynchronous (non-blocking) interface, used by the hybrid pipeline
    def submit(self, x=None):
        """Hand a frame to the core's worker thread and return immediately."""
        if x is not None:
            xa = np.ascontiguousarray(x, dtype=np.float32)
            self.lib.fc_set_input(self.h, self.frame,
                                  xa.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        self.lib.fc_submit(self.h, self.frame)
        idx = self.frame
        self.frame += 1
        return idx

    def wait(self, idx):
        """Block until the frame `idx` has been computed, and return it."""
        lat = ctypes.c_double(0.0)
        rc = self.lib.fc_wait(self.h, idx, ctypes.byref(lat))
        if rc != 0:
            raise RuntimeError("fc_wait failed")
        out = np.ctypeslib.as_array(self.lib.fc_last_output(self.h), shape=(64,)).copy()
        return out, lat.value

    def wait_at(self, idx):
        """Wait for frame `idx` and return THAT frame's output and latency."""
        lat = ctypes.c_double(0.0)
        rc = self.lib.fc_wait(self.h, idx, ctypes.byref(lat))
        if rc != 0:
            raise RuntimeError("fc_wait failed")
        buf = (ctypes.c_float * 64)()
        rc2 = self.lib.fc_last_output_at(self.h, idx, buf)
        if rc2 != 0:
            raise RuntimeError("fc_last_output_at failed")
        return np.array(buf, dtype=np.float32).copy(), lat.value

    def pose(self):
        x, y, t = (ctypes.c_double(), ctypes.c_double(), ctypes.c_double())
        self.lib.fc_last_pose(self.h, ctypes.byref(x), ctypes.byref(y), ctypes.byref(t))
        return x.value, y.value, t.value

    def layer_nnz(self):
        out = []
        for i in range(3):
            nnz = ctypes.c_int64(0)
            ni = ctypes.c_int32(0)
            no = ctypes.c_int32(0)
            self.lib.fc_layer_nnz(self.h, i, ctypes.byref(nnz), ctypes.byref(ni), ctypes.byref(no))
            out.append({"layer": i, "nnz": nnz.value, "n_in": ni.value, "n_out": no.value})
        return out

    def close(self):
        if getattr(self, "h", None):
            self.lib.fc_destroy(self.h)
            self.h = None


if __name__ == "__main__":
    build()
    print("built", LIB_PATH)
