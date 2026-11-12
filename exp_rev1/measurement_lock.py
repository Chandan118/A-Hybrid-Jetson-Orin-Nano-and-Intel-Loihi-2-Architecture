"""A lock that stops CPU-heavy tooling from running during a measurement.

A power measurement is only attributable to the configuration under test if
nothing else is drawing from the same rails.  The quiescence guard
(`quiescence.py`) checks the machine state *before* each window, but it
cannot protect a window that is already open: any process that starts
mid-window contaminates the very numbers that window is producing.

That is not hypothetical.  On 2026-11-05 the 30 Hz matched-rate pass was
aborted at C1_rep1 because `make_paper_figures.py` -- run by the
author's own `verify_final.py` -- was rendering figures at 81 % CPU
inside the measurement window.  The guard caught it and refused to emit
the figure, which is the correct behaviour, but the run was still lost.

So the measurement scripts take this lock, and the tooling checks it.

Usage
-----
    from measurement_lock import measurement_lock

    with measurement_lock("run1_30hz"):
        ...the measurement...

The lock is an exclusive `flock` on `exp_rev1/.measurement.lock`, so it is
released automatically if the process dies -- a stale lock file cannot
wedge the machine.

The tooling side is `tooling_is_blocked()`, which returns a human-readable
reason when a measurement is in progress.
"""

import errno
import fcntl
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK_PATH = os.path.join(HERE, ".measurement.lock")

# The tooling that must not run inside a measurement window.
BLOCKED_TOOLING = {
    "make_paper_figures.py": "renders figures and burns CPU",
    "make_review_response.py": "regenerates the review response and "
                                "re-reads the result files",
    "make_appeal_letters.py": "regenerates the appeal letters",
    "sync_published_numbers.py": "regenerates the README and the letters",
    "verify_final.py": "re-renders the figures as part of its "
                       "reproducibility check",
    "build_manuscript_tables.py": "rebuilds the manuscript tables",
}


class measurement_lock:
    """Exclusive, automatically-released lock held for the duration of a
    measurement run."""

    def __init__(self, name="measurement", path=None):
        self.name = name
        self.path = path or LOCK_PATH
        self._fh = None
        self._fd = None
        self._held = False

    def __enter__(self):
        # NOTE: do NOT open with mode "w" -- that truncates the file, and a
        # second process would wipe the holder's details out of the lock
        # file before it discovers it cannot have the lock.  Open without
        # truncating and write the details only after the flock succeeds.
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        self._fh = os.fdopen(self._fd, "r+")
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            os.close(self._fd)
            self._fd = None
            self._fh = None
            raise SystemExit(
                f"another measurement already holds {self.path} -- "
                f"refusing to start a second one, because two concurrent "
                f"measurements would contaminate each other.")
        self._held = True
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(f"{self.name}\npid={os.getpid()}\n"
                        f"started={time.time()}\n")
        self._fh.flush()
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None
                self._fh = None
        self._held = False
        return False


def _holder():
    """Who holds the lock, if anyone.  Returns None when it is free."""
    if not os.path.exists(LOCK_PATH):
        return None
    try:
        fh = open(LOCK_PATH, "r")
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return fh.read().strip() or "a measurement (details unreadable)"
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return None
    finally:
        fh.close()


def tooling_is_blocked(tool_name=None):
    """Return a reason string if CPU-heavy tooling must not run now.

    Returns None when it is safe to run.
    """
    who = _holder()
    if who is None:
        return None
    why = BLOCKED_TOOLING.get(tool_name, "this tool is CPU-heavy")
    return (f"A MEASUREMENT IS IN PROGRESS ({who}). {tool_name or 'This tool'} "
            f"must not run now, because {why} and it would contaminate the "
            f"measurement window. Wait for the run to finish, or stop it "
            f"first. The measurement lock is {LOCK_PATH}.")


def refuse_if_measurement_running(tool_name=None):
    """Abort with a clear message if a measurement is in progress."""
    reason = tooling_is_blocked(tool_name)
    if reason:
        raise SystemExit(reason)
