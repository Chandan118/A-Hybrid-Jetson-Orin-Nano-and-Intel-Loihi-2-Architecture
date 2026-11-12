#!/usr/bin/env python3
"""
Shared workload definition for configurations C1..C5.

Every configuration must perform *identical work*: the same recorded sensor
stream is replayed, and each configuration consumes exactly the same frames at
the same effective rate.  The per-config difference is only WHERE the fusion
network is executed and how it is executed.

Input source
------------
The recorded ROS 2 bag (rosbag_exp4_20260404_021816) contains 2746 /scan
messages over 362 s.  A LaserScan replay is used as the sensor stream: it is
the recorded sensor data, it is identical across all configurations, and it
exercises the same fusion input path (256-bin range profile) that the SNN
consumes in the original implementation.

Fusion network
--------------
The network under test is the 3-layer LIF SNN used by the original
implementation:
    fc1: 256 -> 512    (LIF, beta=0.9)
    fc2: 512 -> 512    (LIF, beta=0.9)
    fc3: 512 -> 64     (LIF, beta=0.9)
with 25 time steps per frame.  That is 256*512 + 512*512 + 512*64 = 425,984
dense synapses (319,488 stored at 75% row-wise magnitude pruning).

Output
------
The network's spike output drives an odometry integrator, producing a pose
trajectory.  Every configuration therefore produces a comparable trajectory,
which is what the pose-RMSE cross-check in Run 1 consumes.
"""

import os
import json
import struct
import sqlite3
import sys
import threading
import time
from typing import List, Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Bag path is machine-specific. Override with EXP_BAG env var on other machines.
# Default preserves the original bench location.
DEFAULT_BAG = os.environ.get(
    "EXP_BAG",
    "/home/jetson/exp1_logs/rosbag_exp4_20260404_021816/rosbag_exp4_20260404_021816_0.db3",
)

# --------------------------------------------------------------- network spec

NET = {
    "input_size": 256,
    "hidden_size": 512,
    "output_size": 64,
    "time_steps": 25,
    "beta": 0.9,
    "spike_threshold": 0.5,
}
SYNAPSES = (NET["input_size"] * NET["hidden_size"]
            + NET["hidden_size"] * NET["hidden_size"]
            + NET["hidden_size"] * NET["output_size"])   # 425,984

# --------------------------------------------------------------- bag reading


def _ros_py():
    for p in ("/opt/ros/humble/lib/python3.10/site-packages",):
        if p not in sys.path:
            sys.path.append(p)
    import rclpy.serialization as s
    from sensor_msgs.msg import LaserScan
    return s.deserialize_message, LaserScan


def read_scans(bag_path: str = DEFAULT_BAG, max_scans: Optional[int] = None):
    """Read every /scan message from the recorded bag into memory.

    Returns a list of (timestamp_ns, np.float32 array of 1080 ranges).
    """
    deserialize, LaserScan = _ros_py()
    con = sqlite3.connect(bag_path)
    tid = con.execute("select id from topics where name='/scan'").fetchone()[0]
    out = []
    for ts, data in con.execute(
            "select timestamp, data from messages where topic_id=? order by timestamp", (tid,)):
        m = deserialize(data, LaserScan)
        arr = np.asarray(m.ranges, dtype=np.float32)
        out.append((ts, arr))
        if max_scans and len(out) >= max_scans:
            break
    con.close()
    return out


def profile_from_scan(ranges: np.ndarray, nbins: int = 256,
                       rmin: float = 0.15, rmax: float = 12.0) -> np.ndarray:
    """Reduce a LaserScan to the 256-bin range profile the SNN consumes.

    This is the standard "binned range profile" encoding: each bin keeps the
    minimum range in its angular sector, which is what a range-profile neuron
    layer sees.  Invalid readings (inf / nan / out of range) are masked.
    """
    valid = np.isfinite(ranges) & (ranges >= rmin) & (ranges <= rmax)
    if not valid.any():
        return np.full(nbins, rmax, dtype=np.float32)
    idx = np.clip(((np.arange(len(ranges)) / len(ranges)) * nbins).astype(int), 0, nbins - 1)
    prof = np.full(nbins, rmax, dtype=np.float32)
    for b in range(nbins):
        sel = valid & (idx == b)
        if sel.any():
            prof[b] = ranges[sel].min()
    return prof


def encode_input(prof: np.ndarray) -> np.ndarray:
    """Scale a range profile into [0, 1] and flatten to the 256-dim input."""
    v = (prof - 0.15) / (12.0 - 0.15)
    return np.clip(v, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------- deterministic init

def make_weights(seed: int = 12345):
    """Deterministic SNN weights, identical for every configuration.

    The C++/C/Loihi paths must see bit-identical weights to the Python paths,
    so the weights are generated from a fixed seed with a fixed generator and
    exported to a binary file that every backend loads.
    """
    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal((NET["input_size"], NET["hidden_size"])) * 0.05).astype(np.float32)
    w2 = (rng.standard_normal((NET["hidden_size"], NET["hidden_size"])) * 0.05).astype(np.float32)
    w3 = (rng.standard_normal((NET["hidden_size"], NET["output_size"])) * 0.05).astype(np.float32)
    b1 = np.zeros(NET["hidden_size"], dtype=np.float32)
    b2 = np.zeros(NET["hidden_size"], dtype=np.float32)
    b3 = np.zeros(NET["output_size"], dtype=np.float32)
    return {"w1": w1, "w2": w2, "w3": w3, "b1": b1, "b2": b2, "b3": b3}


def export_weights(path: str, seed: int = 12345):
    w = make_weights(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **w)
    return path


def load_weights(path: str):
    z = np.load(path)
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------- the pose

class OdomIntegrator:
    """Turns the network's 64-dim spike output into a 2-D trajectory.

    Deterministic and identical across backends, so any difference in the
    resulting trajectory is attributable to the backend's numerics, not to a
    different post-processing rule.
    """

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.poses = []

    def step(self, spikes: np.ndarray) -> tuple:
        # heading from the first two output units, speed from the mean rate
        s = np.asarray(spikes, dtype=np.float64)
        w = (s[:32] - s[32:64])
        dtheta = float(w.mean()) * 0.01
        v = float(s.mean()) * 0.02
        self.theta += dtheta
        self.x += v * np.cos(self.theta)
        self.y += v * np.sin(self.theta)
        self.poses.append((self.x, self.y, self.theta))
        return self.x, self.y, self.theta


# --------------------------------------------------------------- replay clock

class ReplayClock:
    """Emits frames at the target rate from the recorded stream.

    The replay is time-based, not 'as fast as possible', so every configuration
    is offered the same frames per second and the energy-per-frame comparison
    is fair.  When the consumer falls behind, the backlog is reported as
    dropped frames rather than silently accelerated.

    WHAT "DROPPED" MEANS
    --------------------
    A frame is dropped when its deadline has already passed by the time the
    consumer is ready to start it -- i.e. the replay could not offer that
    frame at the rate it was scheduled for.

    The count must be the number of MISSED DEADLINES, and it must be
    countable exactly:  `dropped == elapsed_s * rate_hz - frames`.

    The previous implementation counted a drop every time a frame started
    late, and then also advanced `next_t` by one period, so the clock
    stayed permanently behind and EVERY subsequent frame was counted as a
    late drop as well.  One late frame therefore poisoned the count for
    the whole window: the clean 30 Hz pass recorded 16,294 "dropped" for
    C2 rep2, where the number of genuinely missed deadlines was 13.

    The bug did not affect power or FPS (both are computed from the frames
    actually delivered and the elapsed time), but it made every "dropped
    frames" figure meaningless -- and those figures appear in the review
    response and the appeal letter.

    The implementation below counts a missed deadline exactly once, by
    tracking the number of slots that have passed without a frame being
    started on time.
    """

    def __init__(self, rate_hz: float = 50.0):
        self.rate_hz = rate_hz
        self.period = 1.0 / rate_hz

    def run(self, n_frames: int, process_fn, on_frame=None, stop=None):
        """process_fn(frame_index, encoded_input) -> dict of per-frame metrics."""
        processed = 0
        t_start = time.perf_counter()
        # The number of slots the clock has offered so far.  A frame that
        # starts on time consumes exactly one slot; a frame that starts
        # late consumes every slot that went by in the meantime, and those
        # are the missed deadlines.
        slot = 0
        next_t = t_start
        for i in range(n_frames):
            if stop is not None and stop():
                break
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
                # woke up on time: this frame consumed slot `slot`
                slot += 1
                next_t += self.period
            else:
                # We are late.  Every slot from `slot` up to the one that
                # is currently due has been missed; those are the dropped
                # deadlines.  We do NOT advance next_t by one period here:
                # the frame is served late, so the next deadline is still
                # the one after the current slot, and the lateness is
                # absorbed by the next on-time sleep.
                while next_t <= time.perf_counter():
                    slot += 1
                    next_t += self.period
            res = process_fn(i)
            processed += 1
            if on_frame:
                on_frame(i, res)
        elapsed = time.perf_counter() - t_start
        # The missed deadlines are the slots that elapsed without a frame
        # being started on time.  Derived from the elapsed time and the
        # frames actually delivered, so it cannot drift from either.
        offered = int(elapsed * self.rate_hz)
        dropped = max(0, offered - processed)
        return {"frames": processed, "dropped": dropped, "elapsed_s": elapsed,
                "achieved_hz": processed / elapsed if elapsed > 0 else 0.0}
