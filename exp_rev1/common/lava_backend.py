#!/usr/bin/env python3
"""
C6 -- the fusion network running through LAVA'S OWN SOFTWARE STACK.

WHAT THIS IS
------------
This configuration executes the SAME fusion network through Intel Lava's
`Loihi2SimCfg` run configuration: Lava's own compiler, Lava's own runtime,
Lava's own process-model framework, with the Loihi 2 process model selected
by Lava's own run configuration.

    IT IS NOT A MEASUREMENT OF Loihi 2 SILICON.  There is no Loihi 2 board
    attached to this bench (see results/interconnect_lookup.json), so this
    measures the Loihi 2 *software stack* running on the Jetson's ARM CPU.
    It is the closest honest link to Loihi 2 that can be measured without
    the hardware, and it is labelled as exactly that everywhere it appears.

WHY IT IS WORTH MEASURING
-------------------------
It is the only configuration in this revision that goes through the SAME
Lava code a Loihi 2 deployment would use.  The question "does the
neuromorphic software path help?" can therefore be answered for the
SOFTWARE path, and the answer is reported as such.

THE TWO VARIANTS
-----------------
`exact` : the reference dense product and the reference LIF recurrence, each
          implemented as a Lava ProcessModel.  The network computes the
          SAME function as C1..C5 and runs through Lava's stack.  This is
          the variant that goes into the energy-per-frame comparison,
          because it isolates the SOFTWARE STACK as the only variable.
`lava`  : Lava's OWN `Dense` and `LIF` processes, exactly as shipped,
          including Lava's one-step dense buffering and its zero-reset LIF.
          This is the variant that measures the stack as a Loihi 2
          deployment would drive it.  Its spike train deviates from the
          reference; the deviation is measured and reported.

IMPLEMENTATION NOTES THAT WERE NECESSARY
----------------------------------------
*   **The Lava compiler must not be run from a directory that contains
    other Python modules.**  `lava.magma.compiler.compiler_graphs` imports
    every module it finds on `sys.path` while looking for ProcessModels,
    and a module that shadows a built-in model makes the compile hang
    forever.  The compile therefore happens in a private empty directory
    (inside the child process, lava_c6_runner.py).
*   **The Lava runtime does not exit on its own.**  `Runtime.stop()` returns
    but the multiprocessing actors keep the process alive, so the backend
    runs the Lava network in a CHILD PROCESS that the parent can terminate.
*   **The Process classes must be top-level classes of a real module.**
    Lava's compiler finds a ProcessModel by searching the module in which
    the Process is defined; see `common/lava_c6_model.py`.
*   **The system Python cannot import `lava.magma`**, and the Lava virtual
    environment has no ROS 2 stack, so the two live in separate processes
    and talk over a pipe.

THIS MODULE RUNS UNDER THE SYSTEM PYTHON and drives the Lava virtual
environment (`/tmp/lavavenv`) as a child process.
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# The Lava virtual environment on this machine.  It is NOT the system
# Python: the system Python has `lava-dl` (the training library) but not
# `lava.magma`; the Lava venv has `lava-nc` but not the ROS 2 stack.
LAVA_VENV = os.environ.get("LAVA_VENV", "/tmp/lavavenv")
LAVA_PY = os.path.join(LAVA_VENV, "bin", "python")
RUNNER = os.path.join(HERE, "lava_c6_runner.py")

# How long the parent waits for the Lava child to reply before giving up.
# The Lava compiler imports every module it finds on sys.path while
# looking for process models, and the runtime does not exit on its own,
# so a broken Lava installation can hang indefinitely.  Without a
# deadline the equivalence test hangs with no diagnostic at all.
LAVA_TIMEOUT_S = float(os.environ.get("LAVA_TIMEOUT_S", "180"))


# --------------------------------------------------------------- environment

def lava_version() -> dict:
    """Which Lava is installed, and which of its parts are usable."""
    out = {"venv": LAVA_VENV, "python": LAVA_PY}
    if not os.path.exists(LAVA_PY):
        out["status"] = "NO LAVA VENV at %s" % LAVA_VENV
        return out
    code = (
        "import importlib, json\n"
        "res = {}\n"
        "for mod in ('lava', 'lava.magma.compiler.compiler',"
        " 'lava.proc.dense.process', 'lava.proc.lif.process',"
        " 'lava.lib.neuromorphic', 'lava.dl'):\n"
        "    try:\n"
        "        m = importlib.import_module(mod)\n"
        "        res[mod] = getattr(m, '__version__', 'present')\n"
        "    except Exception as e:\n"
        "        res[mod] = 'ABSENT: %s' % type(e).__name__\n"
        "try:\n"
        "    from lava.magma.core.run_configs import Loihi2SimCfg\n"
        "    res['Loihi2SimCfg'] = 'present'\n"
        "except Exception:\n"
        "    res['Loihi2SimCfg'] = 'ABSENT'\n"
        "try:\n"
        "    from lava.magma.core.model.nc.model import AbstractNcProcessModel\n"
        "    res['nc_backend'] = 'present'\n"
        "except Exception:\n"
        "    res['nc_backend'] = 'ABSENT (no Loihi 2 code-generation back end)'\n"
        "print(json.dumps(res))\n"
    )
    try:
        r = subprocess.run([LAVA_PY, "-c", code], capture_output=True,
                           text=True, timeout=180, cwd=tempfile.gettempdir())
        out["packages"] = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        out["packages"] = {"error": str(e)}
    try:
        r = subprocess.run([LAVA_PY, "-m", "pip", "show", "lava-nc"],
                           capture_output=True, text=True, timeout=120)
        for line in r.stdout.splitlines():
            if line.lower().startswith("version:"):
                out["lava_nc_version"] = line.split(":", 1)[1].strip()
    except Exception:
        pass
    return out


def lava_available() -> bool:
    """Is Lava's magma (the Loihi 2 process-model framework) importable?"""
    if not os.path.exists(LAVA_PY):
        return False
    try:
        r = subprocess.run(
            [LAVA_PY, "-c",
             "from lava.magma.core.run_configs import Loihi2SimCfg;"
             "from lava.magma.compiler.compiler import Compiler;"
             "print('ok')"],
            capture_output=True, text=True, timeout=120,
            cwd=tempfile.gettempdir())
        return r.stdout.strip().endswith("ok")
    except Exception:
        return False


# ------------------------------------------------------------------- backend

class LavaCpuBackend:
    """C6 -- the fusion network executed by Lava's Loihi 2 CPU simulation.

    The Lava runtime owns its own processes, so the network is run in a child
    process and the parent drives it frame by frame over a pipe.
    """

    name = "C6_lava_loihi2_cpu_simulation"

    def __init__(self, weights, cfg, weights_path=None, variant="exact"):
        self.cfg = dict(cfg)
        self.weights_path = weights_path or os.path.join(
            HERE, "weights_quantised.npz")
        self.variant = variant
        self.steps = int(cfg.get("time_steps", 25))
        self.proc = None
        self.conn = None
        self.info = {}
        self._frame_idx = 0
        self._latencies = []

    # -- lifecycle ---------------------------------------------------------
    def reset(self):
        """Reset the per-frame LIF state.

        Lava's runtime keeps the LIF membrane potential in the process
        model's Var between runs, so the state has to be zeroed explicitly
        before each frame -- exactly as the other backends reset their own
        membrane state.  (The reference network's per-frame state is
        zero-initialised in `reference()` and in every other backend's
        `reset()`.)
        """
        # The zeroing is done by the child, which owns the Var objects; the
        # parent asks for it through the pipe.
        if self.conn is not None:
            self.conn.send(("reset",))
            # the child answers with a "reset_done" tag
            try:
                while True:
                    msg = self.conn.recv()
                    if msg[0] == "reset_done":
                        break
            except Exception:
                pass

    def close(self):
        try:
            if self.conn is not None:
                self.conn.send(("stop",))
                self.conn.close()
        except Exception:
            pass
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=10)
        except Exception:
            pass
        # Withdraw the child's exemption from the contamination watcher.
        try:
            from powermeter import unregister_workload_pid
            if self.proc is not None:
                unregister_workload_pid(self.proc.pid)
        except Exception:
            pass
        self.conn = None
        self.proc = None

    # -- the actual work ---------------------------------------------------
    def _ensure_running(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        import socket
        import pickle

        parent_sock, child_sock = socket.socketpair()
        env = dict(os.environ)
        env["LAVA_C6_WEIGHTS"] = self.weights_path
        env["LAVA_C6_VARIANT"] = self.variant
        env["LAVA_C6_STEPS"] = str(self.steps)
        env["LAVA_C6_FD"] = str(child_sock.fileno())

        # The socket is inherited by the child as a file descriptor; the
        # child's stdout is redirected to a LOG FILE so that its diagnostics
        # can be inspected if the handshake fails (a PIPE would block the
        # child once the buffer filled up).
        log_path = os.path.join(tempfile.gettempdir(),
                                "lava_c6_child_%d.log" % os.getpid())
        self.child_log = log_path
        logf = open(log_path, "w")
        self.proc = subprocess.Popen(
            [LAVA_PY, "-u", RUNNER],
            cwd=tempfile.gettempdir(),
            env=env,
            pass_fds=(child_sock.fileno(),),
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
        logf.close()
        child_sock.close()
        self.sock = parent_sock
        self.conn = _SocketConn(parent_sock, timeout_s=LAVA_TIMEOUT_S)

        # The Lava child IS the workload under test, so it must not be
        # counted as contamination by the power logger's in-window watcher.
        # The exemption is explicit and is withdrawn when the child exits.
        try:
            from powermeter import register_workload_pid
            register_workload_pid(self.proc.pid)
        except Exception:
            pass

        # the child sends exactly one "ready" message before its first frame
        if not _wait_readable(parent_sock, 600):
            out = b""
            try:
                out = self.proc.stdout.read(4096)
            except Exception:
                pass
            self.close()
            raise RuntimeError("the Lava child process did not report back; "
                               "its output was: %r" % out[:2000])
        msg = self.conn.recv()
        if msg[0] != "ready":
            self.close()
            raise RuntimeError("the Lava child process failed to start: %r"
                               % (msg,))
        self.info = msg[1]

    def step(self, x):
        """Evaluate one frame through the Lava runtime.

        The frame is written into the source process's Var, the network is
        run for the frame's 25 time steps, and the output spike train is
        read back from the recorder process's Var.
        """
        self._ensure_running()
        t0 = time.perf_counter()
        self.conn.send(("frame", self._frame_idx,
                        np.asarray(x, dtype=np.float32)))
        self._frame_idx += 1
        kind, idx, spikes = self.conn.recv()
        if kind != "frame":
            raise RuntimeError("unexpected reply from the Lava child: %r"
                               % (kind,))
        dt = (time.perf_counter() - t0) * 1000.0
        self._latencies.append(dt)
        return {"spikes": np.asarray(spikes), "latency_ms": dt}


def _fd3_preexec(fd):
    """Make the pipe available to the child as file descriptor 3."""
    def fn():
        os.dup2(fd, 3)
    return fn


def _wait_readable(sock, timeout_s):
    """Wait until the socket has data to read (or the timeout expires)."""
    import select
    r, _, _ = select.select([sock], [], [], timeout_s)
    return bool(r)


class _SocketConn:
    """A pickle-framed connection over a socket.

    The child and the parent are different Python installations (the system
    Python and the Lava virtual environment), so `multiprocessing.Connection`
    cannot be used directly: it is not a stable wire format across
    interpreters.  A plain socket with pickled frames is.
    """

    def __init__(self, sock, timeout_s=120.0):
        self.sock = sock
        self._buf = b""
        self.timeout_s = timeout_s
        import pickle
        self._pickle = pickle

    def send(self, obj):
        data = self._pickle.dumps(obj, protocol=4)
        self.sock.sendall(len(data).to_bytes(4, "little") + data)

    def _read_exactly(self, n):
        while len(self._buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise EOFError("the Lava child closed the connection")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def recv(self):
        """Receive one frame, with a TIMEOUT.

        The Lava runtime does not exit on its own and the compiler can
        hang while importing modules, so a blocking recv with no
        deadline means a broken Lava installation hangs the whole test
        suite with no diagnostic.  The socket therefore carries a
        timeout, and a timeout is reported as the error it is.
        """
        import struct
        import socket as _socket
        deadline = time.time() + self.timeout_s
        while True:
            self.sock.settimeout(max(0.1, deadline - time.time()))
            try:
                head = self._read_exactly(4)
                break
            except _socket.timeout:
                if time.time() >= deadline:
                    raise RuntimeError(
                        f"the Lava child did not reply within "
                        f"{self.timeout_s:.0f} s -- the Lava runtime or "
                        f"compiler appears to be hung. The child's log is "
                        f"in the temp directory as lava_c6_child_<pid>.log.")
        n = struct.unpack("<I", head)[0]
        return self._pickle.loads(self._read_exactly(n))

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


# ------------------------------------------------------- the reference

def reference_spikes(x, weights, cfg):
    """The float64 reference evaluation of the LIF recurrence.

    This is the DEFINITION of the network, and it is the yardstick against
    which the Lava backend's spike train is checked.
    """
    beta, thr, T = cfg["beta"], cfg["spike_threshold"], cfg["time_steps"]
    w1, w2, w3 = (weights[k].astype(np.float64) for k in ("w1", "w2", "w3"))
    m1 = np.zeros(cfg["hidden_size"], dtype=np.float64)
    m2 = np.zeros(cfg["hidden_size"], dtype=np.float64)
    m3 = np.zeros(cfg["output_size"], dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    out = None
    for _ in range(T):
        m1 = beta * m1 + (x @ w1)
        s1 = (m1 >= thr).astype(np.float64)
        m1 = m1 - thr * s1
        m2 = beta * m2 + (s1 @ w2)
        s2 = (m2 >= thr).astype(np.float64)
        m2 = m2 - thr * s2
        m3 = beta * m3 + (s2 @ w3)
        out = (m3 >= thr).astype(np.float64)
        m3 = m3 - thr * out
    return out


if __name__ == "__main__":
    print(json.dumps(lava_version(), indent=2))
    print("lava.magma available:", lava_available())
