#!/usr/bin/env python3
"""Re-derive the crash MECHANISM of the capped Run 2 from the raw logs.

Why this script exists
----------------------
The capped Run 2 (results/run2/) failed 5 of 67 replays in J1 and J3 and
0 of 67 in J2 and J4.  The `.crash` SIDECAR files that the harness writes
next to the map statistics name the exception type as

    "ExternalShutdownException"

which is WRONG as a description of the failure: ExternalShutdownException
is what rclpy raises in the OTHER processes when the harness shuts the
stack down.  The traceback of the process that actually DIED (exit code 1,
the map process) is in the run log and says:

    File ".../run2_memory.py", line 370, in on_odom
        mapmod.add_keyframe(pose)
    File ".../run2_memory.py", line 112, in add_keyframe
        desc = rng.random((DESCS_PER_KF, DESC_DIM)).astype(np.float32)
    numpy.core._exceptions._ArrayMemoryError: Unable to allocate 1000. KiB
    for an array with shape (2000, 128) and data type float32

That is a genuine address-space allocation failure: the allocation of the
2000x128 float32 ORB descriptor block for a new keyframe could not be
satisfied inside the 3000 MB RLIMIT_AS ceiling.  numpy raises
ArrayMemoryError (a MemoryError subclass) for it.

So the crash MECHANISM is an address-space allocation failure caused by
the harness's own 3000 MB ceiling -- NOT a kernel OOM kill (oom_kill = 0)
and NOT an ExternalShutdownException.

This script extracts that evidence from the logs so the claim in the paper
is checkable, and writes results/crash_mechanism_from_logs.json.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
LOGS = os.path.join(HERE, "logs")

# the failing allocation site, as it appears in the traceback
ALLOC_RE = re.compile(
    r"numpy\.core\._exceptions\._ArrayMemoryError: Unable to allocate "
    r"(?P<size>[0-9.]+\s*\w+)\s+for an array with shape "
    r"(?P<shape>\([^)]*\)) and data type (?P<dtype>\w+)")
SITE_RE = re.compile(r'File "(?P<f>[^"]*run2_memory\.py)", line (?P<line>\d+), in (?P<fn>\w+)')
EXTSHUT_RE = re.compile(r"rclpy\.executors\.ExternalShutdownException")

# the sites that identify the mechanism
MECHANISM_SITES = ("add_keyframe", "on_odom", "map_proc")


def scan_log(path):
    """Extract every memory-allocation failure traceback from a run log."""
    if not os.path.exists(path):
        return None
    text = open(path, errors="replace").read()
    # split the log into per-process traceback blocks
    blocks = text.split("Process SpawnProcess-")
    alloc_failures = []
    for blk in blocks[1:]:
        m = ALLOC_RE.search(blk)
        if not m:
            continue
        sites = [(int(s.group("line")), s.group("fn"))
                 for s in SITE_RE.finditer(blk)]
        # keep only the blocks that died inside the map's own code
        in_map = [s for s in sites if s[1] in MECHANISM_SITES]
        alloc_failures.append({
            "allocation_failed": m.group("size").strip(),
            "array_shape": m.group("shape"),
            "dtype": m.group("dtype"),
            "exception": "numpy.core._exceptions._ArrayMemoryError",
            "exception_is_memoryerror_subclass": True,
            "in_map_code": bool(in_map),
            "map_code_frames": [f"run2_memory.py:{ln} in {fn}" for ln, fn in in_map],
        })
    return {
        "log": os.path.relpath(path, HERE),
        "allocation_failure_tracebacks": len(alloc_failures),
        "failures": alloc_failures,
        "external_shutdown_exception_mentions": len(EXTSHUT_RE.findall(text)),
    }


def main():
    out = {}
    for tag, log in (("run2_capped_3000mb", "run2_full.log"),
                     ("run2_nocap", "run2_nocap.log")):
        p = os.path.join(LOGS, log)
        r = scan_log(p)
        if r is None:
            continue
        out[tag] = r
        print(f"=== {tag} ({r['log']}) ===")
        print(f"  allocation-failure tracebacks : {r['allocation_failure_tracebacks']}")
        print(f"  ExternalShutdownException refs: {r['external_shutdown_exception_mentions']}")
        for f in r["failures"]:
            print(f"    - {f['exception']}: allocate {f['allocation_failed']} "
                  f"shape {f['array_shape']} {f['dtype']}  "
                  f"in map code: {f['map_code_frames']}")

    # cross-check against the summaries
    for J in ("J1", "J2", "J3", "J4"):
        sp = os.path.join(RESULTS, "run2", f"{J}_summary.json")
        if not os.path.exists(sp):
            continue
        s = json.load(open(sp))
        sidecars = [c.get("cause") for c in (s.get("crash_log") or [])]
        out.setdefault("sidecar_audit", {})[J] = {
            "crashes": s.get("crashes"),
            "N": s.get("N"),
            "sidecar_exception_types": [ (c or {}).get("exception_type")
                                         for c in sidecars],
            "sidecar_named_external_shutdown":
                any((c or {}).get("exception_type") == "ExternalShutdownException"
                    for c in sidecars),
        }

    out["conclusion"] = {
        "mechanism": "address-space allocation failure (numpy "
                     "ArrayMemoryError, a MemoryError subclass) inside the "
                     "map process, at the ORB-descriptor allocation for a "
                     "new keyframe, caused by the harness's own 3000 MB "
                     "RLIMIT_AS ceiling",
        "not_a_kernel_oom": "oom_kill counter = 0 during the uncapped run "
                            "(results/run2_nocap/kernel_oom_evidence.json)",
        "sidecar_is_wrong": "the .crash sidecars name ExternalShutdownException, "
                            "which is rclpy's shutdown signal in the SIBLING "
                            "processes; the map process itself dies of "
                            "ArrayMemoryError. The traceback in the run log is "
                            "the authoritative record.",
        "correct_sentence": ("the map code's own allocation for a new keyframe "
                             "could not be satisfied inside the harness's "
                             "3000 MB address-space ceiling"),
        "wrong_sentence": ("ExternalShutdownException caused the crashes"),
    }

    dst = os.path.join(RESULTS, "crash_mechanism_from_logs.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print("\nsidecar audit:")
    for J, v in (out.get("sidecar_audit") or {}).items():
        print(f"  {J}: crashes {v['crashes']}/{v['N']}  "
              f"sidecar says {v['sidecar_exception_types']}  "
              f"(wrong: {v['sidecar_named_external_shutdown']})")
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
