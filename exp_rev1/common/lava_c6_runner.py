#!/usr/bin/env python3
"""
The CHILD side of the Lava backend (C6).

This script is executed by the LAVA VIRTUAL ENVIRONMENT's python
(`/tmp/lavavenv/bin/python`), because it needs `lava.magma`, which the
system Python does not have.

It owns the Lava Runtime and answers frame requests over the socket pair
that the parent handed over:

    in :  ("reset",)                  -- zero the LIF state
          ("frame", idx, float32[256]) -- the encoded frame
    out:  ("ready", info)
          ("reset_done",)
          ("frame", idx, float32[64])  -- the output spike train

WHAT IT IS
----------
The fusion network is executed by Lava's own compiler, Lava's own runtime
and Lava's own `Loihi2SimCfg` run configuration.  Two variants:

  * "exact" -- the reference dense product and the reference LIF recurrence,
    each implemented as a Lava ProcessModel (see lava_c6_model.py).  The
    network therefore computes the SAME function as C1..C5 and runs through
    Lava's software stack.  This is the configuration that goes into the
    energy-per-frame comparison, because it isolates the software stack as
    the only variable.

  * "lava" -- Lava's OWN `Dense` and `LIF` processes, exactly as shipped.
    This measures the Loihi 2 software stack as a Loihi 2 deployment would
    use it, including Lava's one-step dense buffering and its zero-reset
    LIF.  Its spike train deviates from the reference; the deviation is
    measured and reported, not hidden.

WHAT IT IS NOT
--------------
It is NOT a measurement of Loihi 2 SILICON.  No Loihi 2 board is attached
to this bench.  Every number produced here describes Lava's software stack
executing on the Jetson's ARM CPU.
"""

import os
import sys
import shutil
import socket
import tempfile


def main():
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import workload as WL
    from lava.magma.compiler.compiler import Compiler
    from lava.magma.core.run_configs import Loihi2SimCfg
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.runtime.runtime import Runtime
    from lava.magma.core.process.message_interface_enum import ActorType
    import lava_c6_model

    # The socket is handed over as a file descriptor by the parent; its
    # number is given in the environment, so that no fd-renumbering trick
    # (which conflicts with subprocess's own pass_fds handling) is needed.
    fd = int(os.environ["LAVA_C6_FD"])
    conn = socket.socket(fileno=fd)

    import pickle

    def _send(obj):
        data = pickle.dumps(obj, protocol=4)
        conn.sendall(len(data).to_bytes(4, "little") + data)

    def _recv():
        import struct
        def read_exactly(n):
            buf = b""
            while len(buf) < n:
                chunk = conn.recv(65536)
                if not chunk:
                    raise EOFError("the parent closed the connection")
                buf += chunk
            return buf
        n = struct.unpack("<I", read_exactly(4))[0]
        return pickle.loads(read_exactly(n))

    weights_path = os.environ["LAVA_C6_WEIGHTS"]
    variant = os.environ.get("LAVA_C6_VARIANT", "exact")

    weights = WL.load_weights(weights_path)
    cfg = dict(WL.NET)
    cfg["time_steps"] = int(os.environ.get("LAVA_C6_STEPS", cfg["time_steps"]))

    # Lava's Dense has no bias term.  The reference network's biases are all
    # zero, so nothing is lost -- and this is CHECKED, not assumed: the
    # backend refuses to run if a bias is non-zero, because then the result
    # would not be the reference network.
    for bk in ("b1", "b2", "b3"):
        if np.any(weights[bk] != 0):
            raise RuntimeError(
                "the Lava backend does not carry a bias term (Lava's Dense "
                "has none), and the reference network's bias %s is not zero "
                "-- refusing to run, because the result would not be the "
                "reference network." % bk)

    # The Lava compiler imports every module it finds on sys.path while
    # looking for ProcessModels; a stray module in the working directory
    # shadows a built-in model and makes the compile hang forever.  The
    # compile therefore happens in a private, empty temporary directory.
    workdir = tempfile.mkdtemp(prefix="lava_c6_compile_")
    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        src, rec, denses, lifs = lava_c6_model.build_network(
            weights, cfg, variant=variant)
        exe = Compiler().compile(src, Loihi2SimCfg())
    finally:
        os.chdir(cwd)
        shutil.rmtree(workdir, ignore_errors=True)

    rt = Runtime(exe, ActorType.MultiProcessing)
    rt.initialize()
    # The runtime only accepts set_var/get_var once it has been started, so
    # one throw-away step is run to bring it into that state.  Its output is
    # discarded.
    print("CHILD: first step starting", flush=True)
    rt.start(RunSteps(num_steps=1, blocking=True))
    print("CHILD: first step done", flush=True)

    def zero_state():
        """Zero the LIF membrane potentials.

        Lava keeps the LIF state in the process model's Var between runs, so
        the per-frame state has to be zeroed explicitly -- exactly as every
        other backend zeroes its own membrane state in `reset()`.
        """
        for lf in lifs:
            rt.set_var(lf.v.id, np.zeros(int(np.prod(lf.v.shape)),
                                        dtype=np.float32))

    _send(("ready", {"variant": variant,
                     "time_steps": cfg["time_steps"],
                     "layers": 3}))

    while True:
        try:
            msg = _recv()
        except EOFError:
            break
        kind = msg[0]
        if kind == "reset":
            zero_state()
            _send(("reset_done",))
        elif kind == "frame":
            _, idx, x = msg
            print("CHILD: frame %d set_var" % idx, flush=True)
            rt.set_var(src.frame.id, np.asarray(x, dtype=np.float32))
            print("CHILD: frame %d running" % idx, flush=True)
            rt.start(RunSteps(num_steps=cfg["time_steps"], blocking=True))
            print("CHILD: frame %d ran" % idx, flush=True)
            spikes = np.asarray(rt.get_var(rec.last.id)).copy()
            print("CHILD: frame %d got" % idx, flush=True)
            _send(("frame", idx, spikes.astype(np.float32)))
        elif kind == "stop":
            break
        else:
            _send(("error", "unknown request %r" % (kind,)))

    try:
        rt.stop()
    except Exception:
        pass
    # The Lava runtime's multiprocessing actors keep the process alive after
    # stop(); exit explicitly.
    os._exit(0)


if __name__ == "__main__":
    main()
