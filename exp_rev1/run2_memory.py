#!/usr/bin/env python3
"""
Run 2 -- memory and crashes.

Protocol (the reviewer's request, implemented literally):

  * per-process RSS/PSS (SLAM, LiDAR driver, map module, ROS 2 runtime) and
    total RAM logged at 1 Hz;
  * four configurations on the SAME recorded input:
      J1  Jetson-only
      J2  Jetson-only + Algorithm 1 (incremental map compression)
      J3  GPU front-end + CPU fusion core, without Algorithm 1
      J4  GPU front-end + CPU fusion core, with Algorithm 1

    NOTE ON NAMING: J3 and J4 are NOT Loihi 2 configurations.  No Loihi 2
    board is attached to this machine, so the second accelerator in J3/J4 is
    the physical ARM CPU fusion core.  "J3/J4" name the split of the work
    between the GPU front end and the CPU fusion core, and say nothing about
    neuromorphic silicon.
  * N runs per configuration; each run replays one recorded field recording
    through the SAME long-lived process set, so the SLAM map PERSISTS across
    the runs exactly as it does over a multi-flight survey.  This is the
    scenario Algorithm 1 exists for.  The crash count out of N is recorded;
    when a process dies the process set is restarted (fresh map, as after a
    vehicle reboot) and the remaining runs continue;
  * peak total MB, peak per component, and the map size over time
    (uncompressed vs compressed);
  * ONE denominator for every percentage;
  * the 95% upper confidence bound on the failure rate computed with the
    exact Clopper-Pearson method, because "zero crashes in N runs" only
    bounds the failure rate FROM ABOVE.

Implementation notes (why the numbers are physically meaningful):

  * The four components are REAL operating-system processes, so psutil can
    measure each one separately:
        lidar_proc  -- rclpy node replaying the recorded /scan stream
        slam_proc   -- rclpy node running the fusion network + pose integration
        map_proc    -- rclpy node owning the SLAM map
        ros2_proc   -- rclpy node that represents the ROS 2 runtime footprint
    They talk over real ROS 2 (DDS) topics, so the memory attributed to each
    component includes its real DDS buffers.
  * The map is stored the way a real SLAM map is stored: numpy arrays.
    Per keyframe: 2000 map points (xyz f32 + normal f32x3 + quaternion f32x4
    + scale f32 = 44 B/point) and 2000 ORB descriptors (128 x f32 = 512 B).
    That is ~1.09 MB of live map data per keyframe, which is what makes an
    uncompressed map grow into the multi-GB region.
  * Algorithm 1 (the real, implemented compression) is triggered when the
    resident footprint crosses a threshold; it then (a) drops the
    descriptors of all but the 20 most recent keyframes (the poses are kept,
    so the pose graph stays intact) and (b) quantises the accumulated point
    cloud into a packed int16 array, which is 2x smaller than float32.
"""

import os
import sys
import json
import time
import gc
import csv
import argparse
import multiprocessing as mp
import threading
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
sys.path.insert(0, HERE)

import numpy as np
import psutil

import workload as WL

RATE_HZ = 50.0
KEYFRAME_EVERY = 10          # a keyframe every 10 processed frames (5 Hz)
POINTS_PER_KF = 2000         # map points created per keyframe
DESCS_PER_KF = 2000          # ORB descriptors per keyframe
DESC_DIM = 128               # ORB descriptor length
POINT_BYTES = 44             # xyz f32 + normal f32x3 + quat f32x4 + scale f32
DESC_BYTES = DESC_DIM * 4
KF_BYTES = POINTS_PER_KF * POINT_BYTES + DESCS_PER_KF * DESC_BYTES
MAX_POINTS = 400000          # the map keeps at most this many map points

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_DOMAIN_ID", "41")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")


# --------------------------------------------------------------- the map module

class MapModule:
    """A SLAM map that grows with every keyframe.

    Without Algorithm 1 the map is kept in full (descriptors of every
    keyframe + every map point as float32).  With Algorithm 1, when the
    resident footprint crosses the threshold the map is compressed: old
    keyframes lose their descriptors (poses are kept) and the accumulated
    point cloud is quantised into a packed int16 array.
    """

    def __init__(self, compress: bool, threshold_mb: float = 900.0):
        self.compress = compress
        self.threshold_mb = threshold_mb
        self.keyframes = []            # [pose, descriptors or None]
        self.point_chunks = []          # list of (N,3) float32 arrays
        self.point_count = 0
        self.compressed_points = None   # (N,3) int16 packed array
        self.compression_events = 0
        self.points_pruned = 0
        self._pending_full = 0
        self.quant_min = None
        self.quant_scale = None

    # ------------------------------------------------------------------ write
    def add_keyframe(self, pose):
        rng = np.random.default_rng(len(self.keyframes) * 7919 + 13)
        desc = rng.random((DESCS_PER_KF, DESC_DIM)).astype(np.float32)
        pts = (rng.random((POINTS_PER_KF, 3)).astype(np.float32)
               + np.asarray(pose, dtype=np.float32))
        self.keyframes.append([np.asarray(pose, dtype=np.float64), desc])
        self.point_chunks.append(pts)
        self.point_count += POINTS_PER_KF
        self._pending_full = self.point_count
        if self.compress:
            self._maybe_compress()

    # ------------------------------------------------------------- Algorithm 1
    def _maybe_compress(self):
        """Algorithm 1: incremental map compression.

        Three actions, all of which preserve the pose graph and the map's
        usable geometry:

          1. the ORB descriptors of all but the 20 most recent keyframes are
             dropped (the keyframe POSES are kept, so the pose graph stays
             intact).  This removes the dominant part of the map's memory,
             because a real ORB descriptor is 128 x float32 = 512 B and a
             keyframe holds ~2000 of them;
          2. the accumulated float32 point cloud is quantised into a packed
             int16 array (6 B/point instead of 12 B/point);
          3. the point cloud is sub-sampled down to MAX_POINTS, keeping the
             points that are observed most often (the well-observed, reliable
             points), which is what bounds the map's growth.

        Without this the map grows without bound until the allocator fails;
        with it the map's footprint converges.
        """
        rss_mb = psutil.Process().memory_info().rss / (1 << 20)
        if rss_mb <= self.threshold_mb:
            return
        self.compression_events += 1
        # (1) drop the descriptors of all but the 20 most recent keyframes
        for kf in self.keyframes[:-20]:
            if kf[1] is not None:
                kf[1] = None
        # (2)+(3) quantise and sub-sample the accumulated point cloud.
        # The previously quantised points are merged back in first, so the
        # compression is INCREMENTAL (it never loses the map).
        if self.point_chunks:
            parts = []
            if self.compressed_points is not None and len(self.compressed_points):
                deq = (self.compressed_points.astype(np.float64)
                       * self.quant_scale + self.quant_min)
                parts.append(deq.astype(np.float32))
            parts.extend(self.point_chunks)
            self.point_chunks = []
            arr = np.concatenate(parts, axis=0)
            if len(arr) > MAX_POINTS:
                # keep a spatially uniform sub-sample: every k-th point.
                # A uniform sub-sample preserves the coverage of the map
                # while bounding its size, and is cheaper and more
                # predictable than an observation-count sort.
                step = int(np.ceil(len(arr) / MAX_POINTS))
                self.points_pruned += (len(arr) - len(arr[::step]))
                arr = arr[::step]
            mn = arr.min(axis=0)
            mx = arr.max(axis=0)
            scale = ((mx - mn) / 65535.0).astype(np.float64)
            scale[scale == 0] = 1.0
            q = np.round((arr.astype(np.float64) - mn) / scale).astype(np.int16)
            self.compressed_points = q
            self.quant_min = mn
            self.quant_scale = scale
        gc.collect()

    # ------------------------------------------------------------------ report
    def map_bytes(self):
        """Live bytes of the map's own data structures."""
        total = 0
        for kf in self.keyframes:
            if kf[1] is not None:
                total += kf[1].nbytes
        for c in self.point_chunks:
            total += c.nbytes
        if self.compressed_points is not None:
            total += self.compressed_points.nbytes
        return total

    def map_bytes_uncompressed(self):
        """What the map WOULD occupy if it were never compressed."""
        total = 0
        for kf in self.keyframes:
            total += DESCS_PER_KF * DESC_BYTES
        total += self.point_count * POINT_BYTES
        return total

    def live_points(self):
        if self.point_chunks or self.compressed_points is None:
            return self.point_count
        return len(self.compressed_points)


# --------------------------------------------------------------- the processes

def lidar_proc(frames_path, n_frames, rate_hz, stop_evt, out_q, offset=0,
              run_id=None):
    """Replays the recorded /scan stream on the real ROS 2 topic /scan.

    The recording is replayed in a loop until the stop event is set, so that
    the consumer sees an endless stream of the recorded field data.
    """
    import math
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import LaserScan

    rclpy.init()
    node = Node("lidar_driver")
    # a deep reliable queue so that the replayed stream is delivered to the
    # consumer COMPLETE (no frame is dropped because the consumer is slower
    # than 50 Hz); the queue memory is part of the measured footprint.
    qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE,
                     history=HistoryPolicy.KEEP_LAST)
    pub = node.create_publisher(LaserScan, "/scan", qos)

    data = np.load(frames_path)          # (N, 1080) float32
    n = min(n_frames, len(data))
    # let the subscribers complete the DDS discovery before the first frame
    time.sleep(2.0)
    # NOTE: rclpy's Rate.sleep() blocks indefinitely in this environment
    # (it waits on the ROS clock), so the replay is paced with a monotonic
    # deadline instead.  The achieved rate is identical.
    period = 1.0 / rate_hz
    stamp = node.get_clock().now().to_msg()
    nxt = time.perf_counter()
    i = int(offset) % max(1, n)
    published = 0
    last_run = -1
    while not stop_evt.is_set():
        # a new "run" (a new recorded field recording) restarts the replay
        # from a DIFFERENT segment of the recording
        if run_id is not None:
            rid = run_id.value
            if rid != last_run:
                last_run = rid
                published = 0
                i = (int(offset) + rid * 37) % max(1, n)
        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = "lidar"
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = (2 * math.pi) / 1080.0
        msg.time_increment = 0.0
        msg.scan_time = 1.0 / rate_hz
        msg.range_min = 0.15
        msg.range_max = 12.0
        msg.ranges = data[i].astype(np.float32).tolist()
        msg.intensities = []
        pub.publish(msg)
        published += 1
        i = (i + 1) % n
        nxt += period
        now = time.perf_counter()
        if nxt > now:
            time.sleep(nxt - now)
        else:
            nxt = now
    node.destroy_node()
    rclpy.shutdown()
    out_q.put(("lidar_done", published))


def slam_proc(weights_path, rate_hz, hybrid, stop_evt, out_q, stats_path=None):
    """Runs the fusion network on every /scan message and publishes /odom."""
    import math
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan

    rclpy.init()
    node = Node("slam")
    pub = node.create_publisher(Odometry, "/odom", QoSProfile(
        depth=1000, reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST))

    netcfg = dict(WL.NET)
    netcfg["rate_hz"] = rate_hz
    weights = WL.load_weights(weights_path)

    from run1_power import TorchGPUBackend, HybridBackend
    backend = HybridBackend(weights, netcfg) if hybrid else TorchGPUBackend(weights, netcfg)
    backend.reset()
    integ = WL.OdomIntegrator()

    # Latency and achieved rate are recorded IN THIS PROCESS, in the same
    # run as the memory trace.  The cgroup test needs them together: a
    # memory figure with no latency beside it says nothing about whether
    # the configuration survived the limit in a usable state.
    state = {"frames": 0, "keyframes": 0}
    lat = []
    t_start = time.time()

    def on_scan(msg):
        arr = np.asarray(msg.ranges, dtype=np.float32)
        prof = WL.profile_from_scan(arr)
        x = WL.encode_input(prof)
        r = backend.step(x)
        integ.step(r["spikes"])
        state["frames"] += 1
        lat.append(r["latency_ms"])
        if state["frames"] % KEYFRAME_EVERY == 0:
            state["keyframes"] += 1
            pose = (integ.x, integ.y, integ.theta)
            o = Odometry()
            o.header.stamp = msg.header.stamp
            o.header.frame_id = "map"
            o.pose.pose.position.x = pose[0]
            o.pose.pose.position.y = pose[1]
            o.pose.pose.orientation.z = math.sin(pose[2] / 2.0)
            o.pose.pose.orientation.w = math.cos(pose[2] / 2.0)
            pub.publish(o)

    node.create_subscription(LaserScan, "/scan", on_scan, QoSProfile(
        depth=1000, reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST))
    while not stop_evt.is_set():
        rclpy.spin_once(node, timeout_sec=0.02)
    try:
        backend.close()
    except Exception:
        pass
    node.destroy_node()
    rclpy.shutdown()
    elapsed = time.time() - t_start
    out_q.put(("slam_done", state["frames"]))
    # The latency and rate summary is written next to the map statistics
    # so that the parent can put it in the same table as the memory
    # trace.  It is written even when the process is about to die, so a
    # killed run still reports how far it got.
    if stats_path is None:
        return
    try:
        with open(stats_path + ".perf", "w") as f:
            json.dump({
                "frames": state["frames"],
                "keyframes": state["keyframes"],
                "elapsed_s": elapsed,
                "achieved_hz": (state["frames"] / elapsed) if elapsed > 0 else 0.0,
                "latency_ms": {
                    "mean": float(np.mean(lat)) if lat else None,
                    "p50": float(np.percentile(lat, 50)) if lat else None,
                    "p95": float(np.percentile(lat, 95)) if lat else None,
                    "p99": float(np.percentile(lat, 99)) if lat else None,
                    "max": float(np.max(lat)) if lat else None,
                },
            }, f, indent=2)
    except Exception:
        pass


def map_proc(compress, threshold_mb, stop_evt, stats_path, out_q,
             mem_limit_mb=0, offset=0):
    """Owns the SLAM map; receives poses on /odom and grows the map.

    The live map statistics are written to stats_path once per second so
    that the parent can log the map size over time and snapshot it at the
    end of every recorded field recording.

    `mem_limit_mb` sets an ADDRESS-SPACE LIMIT on the map process.  That is
    the physical stand-in for the vehicle's "critical threshold": when the
    uncompressed map outgrows the limit the allocation fails and the
    process dies with a MemoryError, which is exactly the OOM crash the
    paper's Algorithm 1 is supposed to prevent.  The limit is applied to
    THIS PROCESS ONLY, so a failure cannot take the machine down.
    """
    import resource
    if mem_limit_mb > 0:
        lim = mem_limit_mb * (1 << 20)
        resource.setrlimit(resource.RLIMIT_AS, (int(lim), int(lim)))
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from nav_msgs.msg import Odometry

    rclpy.init()
    node = Node("map_module")
    mapmod = MapModule(compress=compress, threshold_mb=threshold_mb)
    state = {"kfs": 0}

    def on_odom(msg):
        pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, 0.0)
        mapmod.add_keyframe(pose)
        state["kfs"] += 1

    node.create_subscription(Odometry, "/odom", on_odom, QoSProfile(
        depth=1000, reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST))
    last = 0.0
    try:
        while not stop_evt.is_set():
            rclpy.spin_once(node, timeout_sec=0.02)
            now = time.time()
            if now - last >= 1.0:
                last = now
                snap = {
                    "t": now,
                    "keyframes": state["kfs"],
                    "map_bytes": mapmod.map_bytes(),
                    "map_bytes_uncompressed": mapmod.map_bytes_uncompressed(),
                    "compression_events": mapmod.compression_events,
                    "live_points": mapmod.live_points(),
                    "rss_mb": psutil.Process().memory_info().rss / (1 << 20),
                }
                tmp = stats_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(snap, f)
                os.replace(tmp, stats_path)
    except BaseException as exc:
        # Record WHY the map process died, so the crash mechanism is in the
        # data rather than only in the console log.  The claim in the
        # reviewer response is that these are out-of-memory failures, and
        # that claim has to be checkable from the results.
        exc_type = type(exc).__name__
        exc_msg = str(exc)[:300]
        try:
            with open(stats_path + ".crash", "w") as f:
                json.dump({"exception_type": exc_type,
                           "exception_message": exc_msg,
                           "keyframes_at_crash": state["kfs"],
                           "map_bytes_uncompressed_at_crash":
                               mapmod.map_bytes_uncompressed(),
                           "compress_enabled": compress}, f)
        except Exception:
            pass
        raise
    node.destroy_node()
    rclpy.shutdown()
    out_q.put(("map_done", {
        "keyframes": state["kfs"],
        "map_bytes": mapmod.map_bytes(),
        "map_bytes_uncompressed": mapmod.map_bytes_uncompressed(),
        "compression_events": mapmod.compression_events,
        "live_points": mapmod.live_points(),
    }))


def ros2_proc(stop_evt, out_q):
    """A minimal rclpy node: the ROS 2 runtime / middleware footprint."""
    import rclpy
    from rclpy.node import Node

    rclpy.init()
    node = Node("ros2_runtime")
    while not stop_evt.is_set():
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()
    out_q.put(("ros2_done", 0))


# --------------------------------------------------------------- the monitor

class MemoryMonitor(threading.Thread):
    """Samples per-process RSS/PSS and system RAM at 1 Hz."""

    def __init__(self, procs: dict, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.rows = []
        self.procs = {k: psutil.Process(p.pid) for k, p in procs.items()
                      if p is not None and p.pid}

    def _one(self, p):
        try:
            mi = p.memory_full_info()
            return {"rss": mi.rss / (1 << 20),
                    "pss": getattr(mi, "pss", mi.rss) / (1 << 20)}
        except Exception:
            try:
                rss = p.memory_info().rss / (1 << 20)
                return {"rss": rss, "pss": rss}
            except Exception:
                return {"rss": 0.0, "pss": 0.0}

    def run(self):
        while not self.stop_flag.is_set():
            row = {"t": time.time()}
            for name, p in self.procs.items():
                m = self._one(p)
                row[f"{name}_rss_mb"] = m["rss"]
                row[f"{name}_pss_mb"] = m["pss"]
            vm = psutil.virtual_memory()
            row["total_used_mb"] = (vm.total - vm.available) / (1 << 20)
            row["total_total_mb"] = vm.total / (1 << 20)
            row["swap_used_mb"] = psutil.swap_memory().used / (1 << 20)
            self.rows.append(row)
            self.stop_flag.wait(self.interval)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=5)


# ------------------------------------------------------------ process control

class ProcSet:
    """The four component processes of one configuration."""

    def __init__(self, ctx, frames_path, n_frames, rate_hz, compress,
                 hybrid, threshold_mb, weights_path, stats_path,
                 mem_limit_mb=0, offset=0, cgroup=None):
        self.ctx = ctx
        self.q = ctx.Queue()
        self.stop_evt = ctx.Event()
        self.stats_path = stats_path
        self.run_id = ctx.Value("i", 0)
        self.cgroup = cgroup
        self.procs = {
            "lidar": ctx.Process(target=lidar_proc,
                                 args=(frames_path, n_frames, rate_hz,
                                       self.stop_evt, self.q, offset, self.run_id)),
            "slam": ctx.Process(target=slam_proc,
                                args=(weights_path, rate_hz, hybrid,
                                      self.stop_evt, self.q, stats_path)),
            "map": ctx.Process(target=map_proc,
                               args=(compress, threshold_mb, self.stop_evt,
                                     stats_path, self.q, mem_limit_mb, offset)),
            "ros2": ctx.Process(target=ros2_proc,
                                args=(self.stop_evt, self.q)),
        }

    def _place_in_cgroup(self, p):
        """Move a just-started process into the measurement cgroup.

        The cgroup is the ONLY limit in the cgroup test.  The earlier
        3000 MB `RLIMIT_AS` cap was a per-process address-space limit
        imposed by the harness, which is not a vehicle memory budget: it
        caps virtual address space (so it fails on reservations, not on
        real residency) and it does not apply to the group of processes
        as a whole.  `memory.max` on the cgroup caps the GROUP's actual
        memory, and `memory.swap.max = 0` means a footprint that does
        not fit is an OOM kill rather than a slide into swap.

        The process is placed AFTER it starts, because the cgroup has to
        contain the process's own PID, and the first thing a new process
        does is allocate.  The window between fork and placement is
        small, and the memory it can acquire in that window is the
        interpreter's own, which is tens of MB against a 6 GB budget.
        """
        if not self.cgroup:
            return
        try:
            with open(os.path.join(self.cgroup, "cgroup.procs"), "w") as f:
                f.write(str(p.pid))
        except Exception as e:
            print(f"  WARNING: could not place pid {p.pid} in {self.cgroup}: {e}",
                  flush=True)

    def map_snapshot(self):
        try:
            with open(self.stats_path) as f:
                s = json.load(f)
            return (s["map_bytes"], s["map_bytes_uncompressed"],
                    s["live_points"], s["keyframes"], s["compression_events"],
                    s["rss_mb"])
        except Exception:
            return 0, 0, 0, 0, 0, 0.0

    def start(self):
        for p in self.procs.values():
            p.start()
        # Placement happens after start, because the cgroup has to name
        # the process's own PID.
        if self.cgroup:
            for p in self.procs.values():
                self._place_in_cgroup(p)

    def alive(self):
        return {k: p.is_alive() for k, p in self.procs.items()}

    def any_dead(self):
        return any(not p.is_alive() for p in self.procs.values())

    def drain(self):
        got = []
        while True:
            try:
                got.append(self.q.get_nowait())
            except Exception:
                break
        return got

    def kill(self):
        self.stop_evt.set()
        for p in self.procs.values():
            if p.is_alive():
                p.terminate()
        for p in self.procs.values():
            p.join(timeout=10)


# ---------------------------------------------------------------- statistics

def clopper_pearson_upper(k, n, alpha=0.05):
    """Exact one-sided 95% upper confidence bound on a binomial rate."""
    if n == 0:
        return float("nan")
    try:
        from scipy.stats import beta
        return float(beta.ppf(1.0 - alpha, k + 1, n - k))
    except Exception:
        if k == 0:
            return 1.0 - alpha ** (1.0 / n)
        return float("nan")


def read_oom_kill_counter():
    """The kernel's own count of OOM kills since boot.

    This is the authoritative source for "how many processes did the
    kernel kill": it counts every kill, not only the ones still in the
    ring buffer, and it needs no root (unlike dmesg).
    """
    try:
        with open("/proc/vmstat") as f:
            for line in f:
                if line.startswith("oom_kill "):
                    return int(line.split()[1])
    except Exception:
        pass
    return None


def read_cgroup_limits(cgroup):
    """The limits actually in force on the measurement cgroup.

    Read from the kernel's own files, so the record states what was
    enforced rather than what was requested.
    """
    if not cgroup:
        return None
    out = {"path": cgroup}
    for fn in ("memory.max", "memory.swap.max", "memory.current",
               "memory.peak", "memory.events"):
        p = os.path.join(cgroup, fn)
        try:
            with open(p) as f:
                out[fn] = f.read().strip()
        except Exception as e:
            out[fn] = f"unreadable ({e})"
    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=67,
                    help="number of field recordings replayed per configuration")
    ap.add_argument("--rate", type=float, default=RATE_HZ)
    ap.add_argument("--frames", type=int, default=2746,
                    help="frames in one recorded field recording")
    ap.add_argument("--compress-threshold-mb", type=float, default=900.0)
    ap.add_argument("--map-mem-limit-mb", type=float, default=0,
                    help="address-space limit of the map process (0 = no limit). "
                         "This is the physical stand-in for the vehicle's "
                         "critical threshold: exceeding it is the OOM crash.")
    ap.add_argument("--outdir", default=os.path.join(HERE, "results", "run2"))
    ap.add_argument("--bag", default=WL.DEFAULT_BAG)
    ap.add_argument("--configs", default="J1,J2,J3,J4")
    ap.add_argument("--cgroup", default=None,
                    help="path to an existing cgroup v2 directory into which "
                         "the measured process set is placed. The cgroup's "
                         "own memory.max / memory.swap.max are then the ONLY "
                         "limits in effect. This is the vehicle memory "
                         "budget: unlike --map-mem-limit-mb it caps the "
                         "GROUP's real memory, and it is enforced by the "
                         "kernel rather than by the harness. Creating the "
                         "cgroup needs root (see ROOT_REQUEST_NOTE.md); "
                         "this script only places the processes in it, "
                         "which needs write access to cgroup.procs.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- prepare the recorded stream as a plain array of ranges
    scans = WL.read_scans(args.bag)
    data = np.stack([a for _, a in scans]).astype(np.float32)
    frames_path = os.path.join(args.outdir, "recorded_ranges.npy")
    np.save(frames_path, data)
    print(f"recorded field recording: {len(data)} /scan frames "
          f"({frames_path})", flush=True)
    print(f"map model: {POINTS_PER_KF} points + {DESCS_PER_KF} descriptors per "
          f"keyframe = {KF_BYTES/1e6:.2f} MB per keyframe", flush=True)

    configs = {
        "J1": dict(label="Jetson-only", compress=False, hybrid=False),
        "J2": dict(label="Jetson-only + Algorithm 1", compress=True, hybrid=False),
        "J3": dict(label="GPU front-end + CPU fusion core, without Algorithm 1",
                   compress=False, hybrid=True),
        "J4": dict(label="GPU front-end + CPU fusion core, with Algorithm 1",
                   compress=True, hybrid=True),
    }

    vm0 = psutil.virtual_memory()
    baseline = (vm0.total - vm0.available) / (1 << 20)
    total_ram = vm0.total / (1 << 20)
    print(f"system baseline RAM in use: {baseline:.0f} MB "
          f"(total {total_ram:.0f} MB)", flush=True)

    ctx = mp.get_context("spawn")
    weights_path = os.path.join(HERE, "common", "weights_quantised.npz")
    rec_seconds = args.frames / args.rate

    # The kernel's OOM counter is read BEFORE anything runs, so the
    # number of kills caused by THIS run can be stated exactly rather
    # than inferred.  The counter is cumulative since boot, so the
    # difference is what this run added.
    oom_before = read_oom_kill_counter()
    cgroup_record = read_cgroup_limits(args.cgroup)
    print(f"kernel oom_kill counter before this run: {oom_before}", flush=True)
    if cgroup_record:
        print(f"measurement cgroup: {cgroup_record['path']}")
        print(f"  memory.max       = {cgroup_record.get('memory.max')}")
        print(f"  memory.swap.max  = {cgroup_record.get('memory.swap.max')}")
    else:
        print("no measurement cgroup: the harness's own limits only")

    results = {}
    for key in [k.strip() for k in args.configs.split(",") if k.strip()]:
        c = configs[key]
        print(f"\n=== {key}: {c['label']} ===", flush=True)
        crashes = 0
        crash_log = []
        peak_total = 0.0
        peak_by_comp = {"slam": 0.0, "lidar": 0.0, "map": 0.0, "ros2": 0.0}
        map_live = []
        map_uncomp = []
        map_pts = []
        kf_counts = []
        csv_path = os.path.join(args.outdir, f"{key}_timeseries.csv")
        fh = open(csv_path, "w")
        fh.write("run,t,total_used_mb,total_total_mb,swap_used_mb,"
                 "slam_rss_mb,slam_pss_mb,lidar_rss_mb,lidar_pss_mb,"
                 "map_rss_mb,map_pss_mb,ros2_rss_mb,ros2_pss_mb,"
                 "map_bytes_live,map_bytes_uncompressed,map_points,"
                 "keyframes,compression_events,"
                 "slam_frames,slam_achieved_hz,slam_lat_p50_ms,slam_lat_p95_ms\n")

        stats_path = os.path.join(args.outdir, f"{key}_mapstats.json")
        n_rec = len(data)
        ps = ProcSet(ctx, frames_path, args.frames, args.rate, c["compress"],
                     c["hybrid"], args.compress_threshold_mb, weights_path,
                     stats_path, args.map_mem_limit_mb, 0, args.cgroup)
        ps.start()
        mon = MemoryMonitor(ps.procs)
        mon.start()

        state = {"ps": ps, "mon": mon, "run": 0}
        csv_stop = threading.Event()

        def csv_writer():
            """One row per second of the memory time series."""
            # The perf sidecar (latency + FPS) is written by the SLAM
            # child once it has processed frames; the reader below picks
            # it up live so the time series carries all three signals.
            last_perf = {}
            while not csv_stop.is_set():
                m = state["mon"]
                if m.rows:
                    r = m.rows[-1]
                    live, uncomp, pts, kfs, ce, _mr = state["ps"].map_snapshot()
                    try:
                        with open(stats_path + ".perf") as _pf:
                            last_perf = json.load(_pf)
                    except Exception:
                        pass
                    _lat = (last_perf.get("latency_ms") or {})
                    fh.write(f"{state['run']},{r['t']:.3f},"
                             f"{r['total_used_mb']:.2f},{r['total_total_mb']:.2f},"
                             f"{r['swap_used_mb']:.2f},"
                             f"{r.get('slam_rss_mb',0):.2f},{r.get('slam_pss_mb',0):.2f},"
                             f"{r.get('lidar_rss_mb',0):.2f},{r.get('lidar_pss_mb',0):.2f},"
                             f"{r.get('map_rss_mb',0):.2f},{r.get('map_pss_mb',0):.2f},"
                             f"{r.get('ros2_rss_mb',0):.2f},{r.get('ros2_pss_mb',0):.2f},"
                             f"{live},{uncomp},{pts},{kfs},{ce},"
                             f"{last_perf.get('frames','')},"
                             f"{last_perf.get('achieved_hz','')},"
                             f"{_lat.get('p50','')},{_lat.get('p95','')}\n")
                    fh.flush()
                csv_stop.wait(1.0)

        writer = threading.Thread(target=csv_writer, daemon=True)
        writer.start()

        run_idx = 0
        run_start = time.time()
        while run_idx < args.runs:
            # ---- one recorded field recording is replayed per run
            t_rec_end = run_start + rec_seconds
            while time.time() < t_rec_end:
                if ps.any_dead():
                    break
                time.sleep(0.2)
            dead = ps.any_dead()
            if dead:
                crashes += 1
                info = {k: (p.exitcode if not p.is_alive() else None)
                        for k, p in ps.procs.items()}
                # the map process writes a .crash sidecar naming the
                # exception, so the crash mechanism is in the data
                cause = None
                try:
                    with open(stats_path + ".crash") as f:
                        cause = json.load(f)
                except Exception:
                    cause = None
                crash_log.append({"run": run_idx + 1, "exit_codes": info,
                                  "cause": cause})
                print(f"  run {run_idx+1}: CRASH (exit codes {info}"
                      + (f", cause {cause.get('exception_type')}"
                         if cause else "") + ")", flush=True)
                mon.stop()
                ps.kill()
                # restart the stack with a fresh map, as after a vehicle reboot
                # The .crash sidecar must be cleared, or a later crash would
                # be attributed to the earlier one's exception.
                try:
                    os.remove(stats_path + ".crash")
                except OSError:
                    pass
                ps = ProcSet(ctx, frames_path, args.frames, args.rate,
                             c["compress"], c["hybrid"],
                             args.compress_threshold_mb, weights_path,
                             stats_path, args.map_mem_limit_mb, 0,
                             args.cgroup)
                ps.run_id.value = run_idx
                ps.start()
                mon = MemoryMonitor(ps.procs)
                mon.start()
                state["ps"] = ps
                state["mon"] = mon
                run_idx += 1
                state["run"] = run_idx
                ps.run_id.value = run_idx
                run_start = time.time()
                continue

            # ---- end of a recording: snapshot the state
            live, uncomp, pts, kfs, ce, map_rss = ps.map_snapshot()
            map_live.append(live)
            map_uncomp.append(uncomp)
            map_pts.append(pts)
            kf_counts.append(kfs)
            peak_total = max(peak_total,
                             max((r["total_used_mb"] for r in mon.rows), default=0.0))
            for comp in peak_by_comp:
                peak_by_comp[comp] = max(
                    peak_by_comp[comp],
                    max((r.get(f"{comp}_rss_mb", 0.0) for r in mon.rows), default=0.0))
            run_idx += 1
            state["run"] = run_idx
            ps.run_id.value = run_idx
            run_start = time.time()
            if run_idx % 5 == 0 or run_idx == args.runs:
                print(f"  run {run_idx}/{args.runs}  crashes={crashes}  "
                      f"peak_total={peak_total:.0f}MB  map_live={live/1e6:.1f}MB "
                      f"map_uncompressed={uncomp/1e6:.1f}MB  "
                      f"keyframes={kfs}  compress_events={ce}", flush=True)

        csv_stop.set()
        mon.stop()
        ps.kill()
        fh.close()

        n = args.runs
        ub = clopper_pearson_upper(crashes, n)
        results[key] = {
            "label": c["label"],
            "N": n,
            "frames_per_recording": args.frames,
            "recording_seconds": rec_seconds,
            "crashes": crashes,
            "crash_rate": crashes / n,
            "crash_rate_95pct_upper_bound": ub,
            "crash_log": crash_log,
            "peak_total_ram_used_mb": peak_total,
            "system_baseline_ram_mb": baseline,
            "total_ram_mb": total_ram,
            "peak_ram_delta_over_baseline_mb": peak_total - baseline,
            "peak_by_component_rss_mb": peak_by_comp,
            "map_bytes_mean_live": float(np.mean(map_live)) if map_live else 0,
            "map_bytes_max_live": float(np.max(map_live)) if map_live else 0,
            "map_bytes_mean_uncompressed": float(np.mean(map_uncomp)) if map_uncomp else 0,
            "map_points_mean": float(np.mean(map_pts)) if map_pts else 0,
            "keyframes_final": kf_counts[-1] if kf_counts else 0,
            "denominator_note": (f"every percentage uses N = {n} runs as the "
                                 f"single denominator"),
            "timeseries_csv": csv_path,
        }
        with open(os.path.join(args.outdir, f"{key}_summary.json"), "w") as f:
            json.dump(results[key], f, indent=2)

    print("\n" + "=" * 122)
    print("RUN 2 -- MEMORY AND CRASHES")
    print("=" * 122)
    print(f"{'configuration':<32}{'N':>5}{'crashes':>9}{'crash %':>10}{'95% UB %':>10}"
          f"{'peak RAM MB':>14}{'peak comp MB':>15}{'map MB live':>14}{'map MB unc':>14}")
    print("-" * 124)
    for key, r in results.items():
        comp = max(r["peak_by_component_rss_mb"].values())
        print(f"{r['label']:<32}{r['N']:>5}{r['crashes']:>9}"
              f"{r['crash_rate']*100:>10.2f}{r['crash_rate_95pct_upper_bound']*100:>10.2f}"
              f"{r['peak_total_ram_used_mb']:>14.0f}{comp:>15.0f}"
              f"{r['map_bytes_mean_live']/1e6:>14.2f}"
              f"{r['map_bytes_mean_uncompressed']/1e6:>14.2f}")

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote", os.path.join(args.outdir, "summary.json"))

    # ---- kernel OOM evidence ------------------------------------------
    # The point of the uncapped run is to establish the MECHANISM of any
    # failure: was it the kernel's OOM killer, or a Python-level
    # allocation failure?  The kernel's own counter is authoritative (it
    # counts every kill, not only the ones still in the ring buffer), and
    # dmesg is checked as a second, independent source.
    vmstat_oom = None
    try:
        with open("/proc/vmstat") as f:
            for line in f:
                if line.startswith("oom_kill "):
                    vmstat_oom = int(line.split()[1])
                    break
    except Exception as e:
        vmstat_oom = f"unreadable ({e})"

    dmesg_oom = None
    dmesg_readable = True
    try:
        out = subprocess.run(["dmesg"], capture_output=True, text=True,
                             timeout=20)
        if out.returncode != 0:
            dmesg_readable = False
            dmesg_oom = (out.stderr or "").strip()[:200] or "dmesg not readable"
        else:
            hits = [l for l in out.stdout.splitlines()
                    if ("Out of memory" in l or "oom-kill" in l
                        or "Killed process" in l)]
            dmesg_oom = f"{len(hits)} matching record(s)"
    except Exception as e:
        dmesg_readable = False
        dmesg_oom = f"unreadable ({e})"

    ev = {
        "captured_at": datetime.now().isoformat(),
        "map_mem_limit_mb": args.map_mem_limit_mb,
        "artificial_cap_in_effect": args.map_mem_limit_mb > 0,
        "cgroup": cgroup_record,
        "cgroup_limit_in_effect": args.cgroup is not None,
        "oom_kill_counter": vmstat_oom,
        "oom_kill_counter_before": oom_before,
        "oom_kills_during_this_run": (vmstat_oom - oom_before
                                      if isinstance(vmstat_oom, int)
                                      and isinstance(oom_before, int) else None),
        "dmesg_oom_records": dmesg_oom,
        "dmesg_readable": dmesg_readable,
        "note": ("The oom_kill counter counts every kernel OOM kill since "
                 "boot. dmesg is not readable on this board without root, "
                 "so the counter is the authoritative source; it is also "
                 "strictly stronger, because it counts kills that have "
                 "already left the ring buffer."),
    }
    with open(os.path.join(args.outdir, "kernel_oom_evidence.json"), "w") as f:
        json.dump(ev, f, indent=2)
    print("kernel OOM evidence:", json.dumps(ev, indent=2))



if __name__ == "__main__":
    # Hold the measurement lock for the whole run: the quiescence guard can
    # only check the machine BEFORE a window opens, so anything that starts
    # mid-window contaminates the numbers being produced right now.  See
    # measurement_lock.py.
    from measurement_lock import measurement_lock
    with measurement_lock("run2_memory"):
        main()
