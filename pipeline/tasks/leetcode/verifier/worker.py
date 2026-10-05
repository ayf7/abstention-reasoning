"""Verifier worker: one spawned process pinned to one CPU.

Started with the `spawn` start method, so it holds no state from the service
(in particular, no expected outputs) and every fork it makes is equally clean.
It receives jobs over a pipe and returns raw sandbox runs; it never judges.
"""

import os

# Modules the dataset preamble and typical solutions import. Loading them here,
# before any fork, keeps import cost out of the measured CPU time and keeps
# the child from needing to read them from disk.
_PRELOAD = [
    "bisect", "collections", "copy", "datetime", "decimal", "fractions", "functools",
    "heapq", "itertools", "math", "operator", "random", "re", "statistics", "string",
    "typing", "array", "json", "sortedcontainers",
]
_THREAD_ENV = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]


def main(conn, cpu: int, memory_mb: int) -> None:
    os.sched_setaffinity(0, {cpu})
    for var in _THREAD_ENV:
        os.environ[var] = "1"
    import importlib
    for name in _PRELOAD:
        importlib.import_module(name)
    from pipeline.tasks.leetcode.verifier.sandbox import Sandbox

    sandbox = Sandbox(memory_mb=memory_mb)
    conn.send({"ready": cpu})
    while True:
        try:
            job = conn.recv()
        except EOFError:
            return
        if job is None:
            return
        conn.send(sandbox.run(job))
