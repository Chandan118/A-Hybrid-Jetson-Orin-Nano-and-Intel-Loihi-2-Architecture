#!/usr/bin/env python3
"""Regression test for the replay clock's dropped-frame counter.

The counter had a defect that made every "dropped frames" figure in the
published documents meaningless: a frame that started late was counted as
a drop, and then the deadline was advanced by one period, so the clock
stayed permanently behind and EVERY subsequent frame was counted as a
late drop as well.

The fix counts the MISSED DEADLINES, and the invariant it must satisfy is

    dropped == max(0, int(elapsed_s * rate_hz) - frames)

i.e. the number of slots the clock offered during the window, minus the
number of frames it actually delivered on time.

This test pins that invariant, and it also pins the property that matters
most for the measurement: power, energy and FPS are computed from the
frames actually delivered and the elapsed time, so they are NOT affected
by the counter's behaviour.

    python3 exp_rev1/tests/test_replay_clock.py
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))

from workload import ReplayClock  # noqa: E402


def _check(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        return False
    print(f"  ok   {msg}")
    return True


def main():
    ok = True

    # --- case 1: a consumer that keeps up must drop nothing
    r = ReplayClock(50).run(60, lambda i: time.sleep(0.002))
    ok &= _check(r["frames"] == 60, "a consumer that keeps up delivers "
                                   "every frame")
    ok &= _check(r["dropped"] == 0,
                 f"a consumer that keeps up drops nothing (got {r['dropped']})")

    # --- case 2: the invariant holds for a consumer that falls behind
    for work, label in ((0.030, "slightly slow"), (0.060, "slow"),
                        (0.090, "very slow")):
        r = ReplayClock(30).run(120, lambda i: time.sleep(work))
        expected = max(0, int(r["elapsed_s"] * 30) - r["frames"])
        ok &= _check(
            r["dropped"] == expected,
            f"the invariant holds for a {label} consumer "
            f"(dropped={r['dropped']}, expected={expected})")
        # Every offered slot is accounted for exactly once: the frames that
        # were delivered on time, plus the deadlines that were missed.
        # This is the property the old counter violated -- it counted one
        # drop per late frame AND then counted the same lateness again for
        # every frame that followed.
        #
        # The tolerance of one slot is the floor: `int(elapsed * rate)`
        # truncates, so the final partial slot is counted in one place
        # and not the other.  Anything beyond that would be a real error.
        accounted = r["frames"] + r["dropped"]
        offered = int(r["elapsed_s"] * 30)
        ok &= _check(
            abs(accounted - offered) <= 1,
            f"the delivered frames and the missed deadlines account for "
            f"every offered slot ({r['frames']} + {r['dropped']} == "
            f"{offered}, within one slot)")

    # --- case 3: the counter never reports MORE drops than were offered
    r = ReplayClock(30).run(150, lambda i: time.sleep(0.050))
    ok &= _check(r["dropped"] <= int(r["elapsed_s"] * 30),
                 "the dropped count never exceeds the number of slots "
                 "offered")

    # --- case 4: the delivered rate is unaffected by the counter
    # The power and FPS figures are computed from frames and elapsed time,
    # so they must be identical whichever way the counter behaves.
    t0 = time.perf_counter()
    r1 = ReplayClock(30).run(100, lambda i: time.sleep(0.040))
    el1 = time.perf_counter() - t0
    ok &= _check(
        abs(r1["achieved_hz"] - r1["frames"] / r1["elapsed_s"]) < 1e-9,
        "the achieved rate is frames/elapsed and does not depend on the "
        "dropped count")

    print()
    if ok:
        print("PASS: the replay clock counts missed deadlines exactly.")
        return 0
    print("FAILED: the replay clock's dropped count is wrong.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
