#!/usr/bin/env python3
"""C3 measured INSIDE a real ROS 2 node (Reviewer 1 request).

What this does
--------------
The fair-baseline harness (run1_power.py) measures C3 as a pybind11 C++ core
driven by a Python driver: the Python side pre-encodes every /scan into the
256-bin range profile, then hands the encoded vector to the C++ core.

Reviewer 1 asked for a C++/ROS 2 baseline.  The repository already contains
one: ros2_ws/src/fusion_core_node.  That node is a real ROS 2 node that

  * subscribes to /scan (sensor_msgs/msg/LaserScan),
  * performs the SAME front-end encoding (profile_from_scan + encode_input)
    IN C++ (fusion_core_node.cpp: profile() + fc_set_input()),
  * submits the frame to the int8/sparse-CSR fusion core,
  * publishes the resulting pose on /odom and /fusion_pose.

This script measures exactly that configuration:

  1. launch the node with `ros2 run` (so the real ROS 2 executor, the DDS
     transport and the subscription callback are in the loop),
  2. publish every recorded /scan from the bag into /scan,
  3. CLOSED LOOP: do not publish frame N+1 until the node has published the
     pose for frame N.  This makes the replay lossless -- with a best-effort
     /scan topic an open-loop publisher silently drops frames, which would
     make the throughput number meaningless,
  4. log the board power over the same window with the same PowerLogger
     instrument as every other configuration, and
  5. verify the pose trajectory the node produced against the float64
     reference trajectory, so the ROS 2 path is proven to compute the SAME
     result as the standalone C++ core.

The measurement boundary is IDENTICAL to the fair baseline: the window
covers the replay only.  The front end IS inside the node and therefore
IS inside the measurement window -- unlike the C1/C3/C5 harness, where the
front end is applied by the Python driver BEFORE the window opens.

The output is results/run1_fairbaseline/ros2_c3_summary.json.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))

import workload as WL  # noqa: E402
from powermeter import (PowerLogger, InstrumentMap,  # noqa: E402
                       register_workload_pid)

DEFAULT_BAG = os.environ.get(
    "EXP_BAG",
    "/home/jetson/exp1_logs/rosbag_exp4_20260404_021816/rosbag_exp4_20260404_021816_0.db3",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default=DEFAULT_BAG)
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--rate", type=float, default=50.0,
                    help="target rate; the closed loop will not exceed it")
    ap.add_argument("--outdir", default=os.path.join(HERE, "results", "run1_fairbaseline"))
    ap.add_argument("--logdir", default=os.path.join(HERE, "logs", "run1_fairbaseline"))
    ap.add_argument("--rep", type=int, default=-1,
                    help="repeat index; -1 = single run, >=0 = run the repeat "
                         "and append it to ros2_c3_repeats.json")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    scans = WL.read_scans(args.bag)
    print(f"read {len(scans)} /scan messages")
    scans = scans[: args.frames]
    n_frames = len(scans)

    # ---- pre-flight: no other node may be alive ------------------------
    # A leftover node from a previous run would silently take half the
    # frames and the measurement would be meaningless.  Refuse to run.
    try:
        out = subprocess.run(["pgrep", "-af", "fusion_core_node"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        out = ""
    if out:
        print("REFUSING TO START: another fusion_core_node is already running:")
        print(out)
        print("Kill it first:  pkill -f fusion_core_node")
        raise SystemExit(2)

    # ---- start the node -------------------------------------------------
    node_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source " + os.path.join(HERE, "ros2_ws", "install", "setup.bash") + " && "
        "exec ros2 run fusion_core_node fusion_core_node --ros-args "
        "--params-file " + os.path.join(HERE, "ros2_ws", "src", "fusion_core_node",
                                        "config", "fusion_core_node.yaml")
    )
    env = dict(os.environ)
    env["ROS_DOMAIN_ID"] = os.environ.get("ROS_DOMAIN_ID", "42")
    # start_new_session puts the node in its own process group, so the whole
    # group (the `ros2 run` wrapper AND the node it execs) can be killed.
    # Killing only the wrapper would leave the node running and it would keep
    # competing for /scan with the next run -- which silently splits the
    # frames between two nodes and destroys the measurement.
    node = subprocess.Popen(["bash", "-c", node_cmd], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    # the node IS the workload under test: declare it, so the contamination
    # watcher does not report the measured process as an intruder
    register_workload_pid(node.pid)
    node_log = open(os.path.join(args.logdir, "ros2_c3_node.log"), "w")

    def pump():
        for line in node.stdout:
            node_log.write(line)
            node_log.flush()
    threading.Thread(target=pump, daemon=True).start()

    for _ in range(60):
        time.sleep(0.5)
        if node.poll() is not None:
            print("NODE DIED EARLY")
            raise SystemExit(1)
    time.sleep(2.0)
    print("node up")

    # `ros2 run` is a wrapper that SPAWNS the node as a child process, so the
    # pid we started is NOT the node's pid.  Find the real node pid and
    # register it, otherwise the contamination watcher reports the measured
    # workload itself as an intruder.
    node_pid = None
    try:
        out = subprocess.run(["pgrep", "-f", "install/fusion_core_node/lib/"
                                            "fusion_core_node/fusion_core_node"],
                             capture_output=True, text=True).stdout.split()
        pids = [int(p) for p in out]
        if pids:
            node_pid = pids[0]
            register_workload_pid(node_pid)
            print("node pid", node_pid, "registered as workload")
    except Exception as e:
        print("could not resolve node pid:", e)

    # ---- publisher + subscriber -----------------------------------------
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry

    rclpy.init()
    node_py = Node("c3_ros2_replay")

    # RELIABLE + deep queue on the publisher side.  The node subscribes with
    # SensorDataQoS (best effort, depth 5), which is compatible with a
    # reliable publisher; the deep queue on this side means the publisher
    # itself never drops, and the closed loop below means the subscriber
    # never overruns either.
    pub_qos = QoSProfile(depth=200,
                          history=QoSHistoryPolicy.KEEP_LAST,
                          reliability=QoSReliabilityPolicy.RELIABLE)
    pub = node_py.create_publisher(LaserScan, "scan", pub_qos)

    poses = []
    pose_lock = threading.Lock()

    def on_odom(msg):
        with pose_lock:
            poses.append((msg.pose.pose.position.x,
                          msg.pose.pose.position.y,
                          msg.pose.pose.orientation.z,
                          msg.pose.pose.orientation.w))

    sub_qos = QoSProfile(depth=500,
                          history=QoSHistoryPolicy.KEEP_LAST,
                          reliability=QoSReliabilityPolicy.RELIABLE)
    node_py.create_subscription(Odometry, "odom", on_odom, sub_qos)

    # the subscription must be spun from a thread of its own: the publisher
    # loop below blocks on the pose count, so it cannot also spin.
    stop_spin = threading.Event()

    def spin_forever():
        while not stop_spin.is_set():
            rclpy.spin_once(node_py, timeout_sec=0.05)
    threading.Thread(target=spin_forever, daemon=True).start()

    for _ in range(40):
        time.sleep(0.1)
    print("publishers on /scan:", node_py.count_publishers("scan"))

    # ---- power window ---------------------------------------------------
    imap = InstrumentMap(jetson_ina="/sys/class/hwmon/hwmon1")
    csv_path = os.path.join(args.logdir,
                            "ros2_c3.csv" if args.rep < 0 else f"ros2_c3_rep{args.rep}.csv")
    logger = PowerLogger(csv_path, rate_hz=20.0, imap=imap,
                         tegrastats_interval_ms=100)
    # the logger owns the window: it covers the whole replay
    log_thread = threading.Thread(
        target=lambda: logger.run(n_frames / args.rate + 15.0,
                                  label="C3_ros2_node"), daemon=True)
    log_thread.start()
    time.sleep(1.0)

    # ---- closed-loop replay ---------------------------------------------
    period = 1.0 / args.rate
    t_start = time.perf_counter()
    published = 0
    lost = 0
    for ts, arr in scans:
        target = t_start + published * period
        now = time.perf_counter()
        if target > now:
            time.sleep(target - now)

        msg = LaserScan()
        msg.header.stamp.sec = int(ts // 10**9)
        msg.header.stamp.nanosec = int(ts % 10**9)
        msg.header.frame_id = "laser"
        msg.angle_min = float(np.arctan2(arr[0], arr[len(arr) // 2]))
        msg.angle_max = float(np.arctan2(arr[-1], arr[len(arr) // 2]))
        msg.angle_increment = 0.001
        msg.time_increment = 0.0
        msg.scan_time = period
        msg.range_min = 0.15
        msg.range_max = 12.0
        msg.ranges = [float(x) for x in arr]
        pub.publish(msg)
        published += 1

        # closed loop: wait for the pose of THIS frame
        deadline = time.perf_counter() + 2.0
        while True:
            with pose_lock:
                n_odom = len(poses)
            if n_odom >= published:
                break
            if time.perf_counter() > deadline:
                lost += 1
                break
            time.sleep(0.0005)

    t_pub = time.perf_counter() - t_start
    print(f"published {published} frames in {t_pub:.2f} s "
          f"({published / t_pub:.2f} Hz), timed-out {lost}")

    # let the node drain
    time.sleep(2.0)
    stop_spin.set()
    log_thread.join(timeout=25.0)

    node.terminate()
    try:
        node.wait(timeout=10)
    except Exception:
        node.kill()
    # kill the whole process group: the `ros2 run` wrapper execs the node, so
    # the node itself is the group leader and this removes it completely.
    try:
        os.killpg(os.getpgid(node.pid), signal.SIGKILL)
    except Exception:
        pass
    node_log.close()

    # ---- results --------------------------------------------------------
    rows = logger.rows
    pw = [r["vdd_in_w"] for r in rows if r.get("vdd_in_w") is not None]
    with pose_lock:
        poses = list(poses)
    frames_processed = len(poses)
    fps = frames_processed / t_pub if t_pub else 0.0

    out = {
        "config": "C3_ros2_node",
        "measurement": ("ros2 node (real executor + DDS) + recorded-bag "
                        "closed-loop replay"),
        "frames_published": published,
        "frames_processed_by_node": frames_processed,
        "closed_loop_timeouts": lost,
        "elapsed_s": t_pub,
        "fps": fps,
        "sustains_50hz": fps >= 49.9,
        "mean_power_w": float(np.mean(pw)) if pw else None,
        "std_power_w": float(np.std(pw)) if pw else None,
        "ina3221_vdd_in_mean_w": float(np.mean(pw)) if pw else None,
        "ina3221_vdd_in_std_w": float(np.std(pw)) if pw else None,
        "power_csv": csv_path,
        "node_log": os.path.join(args.logdir, "ros2_c3_node.log"),
        "contamination": logger.contamination_record(),
        "note": ("The front end (profile_from_scan + encode_input) is implemented "
                 "IN C++ INSIDE the node, so it IS inside the measurement window. "
                 "This is a single configuration, not a 3-repeat fair baseline."),
    }
    if poses:
        out["trajectory_points"] = len(poses)
        out["pose_final_xy"] = [poses[-1][0], poses[-1][1]]
        np.save(os.path.join(args.outdir, "ros2_c3_poses.npy"),
                np.array([[a, b, c, d] for a, b, c, d in poses], dtype=np.float64))

    # ---- trajectory check against the float64 reference ------------------
    try:
        sys.path.insert(0, HERE)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run1c_pose_check", os.path.join(HERE, "run1c_pose_check.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        enc = np.stack([WL.encode_input(WL.profile_from_scan(a)) for _, a in scans])
        cfg = dict(WL.NET)
        cfg["_w"] = WL.load_weights(os.path.join(HERE, "common", "weights_quantised.npz"))
        ref = mod.reference_trajectory(enc, cfg, len(enc))
        p = np.array([[a, b] for a, b, _, _ in poses], dtype=np.float64)
        rmse, n = mod.pose_rmse(p, ref)
        out["pose_rmse_vs_float64_reference_m"] = rmse
        out["pose_rmse_frames_compared"] = n
        out["reference_final_xy"] = [float(ref[-1, 0]), float(ref[-1, 1])]
    except Exception as e:
        out["pose_check_error"] = repr(e)

    with open(os.path.join(args.outdir, "ros2_c3_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "contamination"},
                     indent=2))

    # ---- append to the repeat file --------------------------------------
    if args.rep >= 0:
        rp = os.path.join(args.outdir, "ros2_c3_repeats.json")
        runs = []
        if os.path.exists(rp):
            runs = json.load(open(rp))
        runs.append({"rep": args.rep,
                     "mean_power_w": out["mean_power_w"],
                     "std_power_w": out["std_power_w"],
                     "fps": out["fps"],
                     "sustains_50hz": out["sustains_50hz"],
                     "frames_processed_by_node": out["frames_processed_by_node"],
                     "closed_loop_timeouts": out["closed_loop_timeouts"],
                     "pose_rmse_vs_float64_reference_m":
                         out.get("pose_rmse_vs_float64_reference_m"),
                     "contamination_clean": out["contamination"]["clean"]})
        with open(rp, "w") as f:
            json.dump(runs, f, indent=2)
        if len(runs) >= args.repeats:
            pw = [r["mean_power_w"] for r in runs]
            fps = [r["fps"] for r in runs]
            agg = {
                "config": "C3_ros2_node",
                "n_repeats": len(runs),
                "power_w_mean": float(np.mean(pw)),
                # SAMPLE sd (ddof = 1), matching the across-repeat
                # convention used by analyse_run1.py, the manuscript
                # table builder and the Table 1 aggregate.  Using the
                # population sd here would make the same repeat set
                # quote a different +/- in a different document.
                "power_w_sd": float(np.std(pw, ddof=1)) if len(pw) > 1 else 0.0,
                "fps_mean": float(np.mean(fps)),
                "fps_sd": float(np.std(fps, ddof=1)) if len(fps) > 1 else 0.0,
                "sustains_50hz_all": all(r["sustains_50hz"] for r in runs),
                "frames_processed_all": [r["frames_processed_by_node"] for r in runs],
                "pose_rmse_vs_float64_reference_m":
                    [r["pose_rmse_vs_float64_reference_m"] for r in runs],
                "contamination_clean_all": [r["contamination_clean"] for r in runs],
            }
            with open(os.path.join(args.outdir, "ros2_c3_aggregate.json"), "w") as f:
                json.dump(agg, f, indent=2)
            print("\n=== ROS 2 C3 aggregate ===")
            print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
