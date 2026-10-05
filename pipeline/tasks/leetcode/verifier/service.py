"""Verifier service: a pool of pinned sandbox workers behind a unix socket.

The service is stateless with respect to problems: each request carries the
problem specs it needs, so RL and evaluation callers share one process without
loading a bank. It caches only gold-solution timings, keyed by spec hash.

    spec = {"setup": str, "entry": str, "tests": [args_src], "expected": [json_str], "gold": str}
    verify(problems={key: spec}, jobs=[(key, code or None)]) -> [result]

A result is `verifier.judge`'s dict plus `limit_s`, `gold_s` and `rechecked`.
`code=None` means the caller found no code block (MALFORMED, nothing runs).

Timing gate: a correct submission passes if its CPU time <= max(k * t_gold,
floor). t_gold is the median of `gold_runs` gold runs on this pool. A correct
submission over the limit is re-run back to back with gold on one worker, and
that pair decides, which cancels clock drift between workers and over time.

Run as a process (never forked from Ray/vLLM):
    python -m pipeline.tasks.leetcode.verifier --socket /path/v.sock --cpus 0,2,4
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from multiprocessing.connection import Client, Listener

from pipeline.tasks.leetcode.verifier import verifier, worker
from pipeline.tasks.leetcode.verifier.verifier import INFRA_ERROR, MALFORMED, PASS, TOO_SLOW

SOCKET_ENV = "VERIFIER_SOCKET"


def physical_cores(cpus=None) -> list[int]:
    """One logical CPU per physical core within `cpus` (default: own affinity).

    A core is used only if all its hyperthread siblings are in the set, so the
    sibling can be left idle and does not perturb timing.
    """
    allowed = set(cpus if cpus is not None else os.sched_getaffinity(0))
    chosen, seen = [], set()
    for cpu in sorted(allowed):
        siblings = hyperthread_siblings(cpu)
        if siblings in seen:
            continue
        seen.add(siblings)
        if siblings <= allowed:
            chosen.append(min(siblings))
    return chosen


def hyperthread_siblings(cpu: int) -> frozenset[int]:
    """All logical CPUs on the same physical core as `cpu`, itself included."""
    with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
        return frozenset(_parse_cpu_list(f.read()))


def _parse_cpu_list(text: str) -> list[int]:
    out = []
    for part in text.strip().split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def spec_key(spec: dict) -> str:
    """Stable hash of everything that affects a problem's verdicts."""
    blob = json.dumps([spec["setup"], spec["entry"], spec["tests"], spec["expected"], spec["gold"]])
    return hashlib.sha1(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


class _Slot:
    """One worker process and the pipe to it. Restarted if it dies or hangs."""

    def __init__(self, ctx, cpu: int, memory_mb: int):
        self.ctx, self.cpu, self.memory_mb = ctx, cpu, memory_mb
        self.restarts = 0
        self._start()

    def _start(self) -> None:
        parent, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=worker.main, args=(child, self.cpu, self.memory_mb), daemon=True)
        self.proc.start()
        child.close()
        self.conn = parent
        if not self.conn.poll(60) or self.conn.recv() != {"ready": self.cpu}:
            raise RuntimeError(f"verifier worker on cpu {self.cpu} failed to start")

    def restart(self) -> None:
        self.restarts += 1
        self.proc.kill()
        self.proc.join()
        self.conn.close()
        self._start()

    def run(self, job: dict):
        """Raw sandbox run, or None if the worker failed (it is then restarted)."""
        try:
            self.conn.send(job)
            # Sandbox.run enforces the deadline itself; the margin covers fork,
            # reaping and pickling. No answer by then means the worker is wedged.
            if self.conn.poll(float(job["timeout_s"]) + 30):
                return self.conn.recv()
        except (EOFError, OSError):
            pass
        self.restart()
        return None

    def close(self) -> None:
        try:
            self.conn.send(None)
        except OSError:
            pass
        self.proc.join(5)
        if self.proc.is_alive():
            self.proc.kill()


class VerifierPool:
    def __init__(self, cpus: list[int], memory_mb: int = 2048, k: float = 3.0, floor_s: float = 0.05,
                 gold_runs: int = 3, timeout_s: float = 10.0):
        self.k, self.floor_s, self.gold_runs, self.timeout_s = k, floor_s, gold_runs, timeout_s
        ctx = mp.get_context("spawn")
        self._slots = [_Slot(ctx, cpu, memory_mb) for cpu in cpus]
        self._tasks: queue.Queue = queue.Queue()
        self._threads = [threading.Thread(target=self._loop, args=(s,), daemon=True) for s in self._slots]
        for t in self._threads:
            t.start()
        self._gold: dict[str, dict] = {}
        self._gold_lock = threading.Lock()

    @property
    def num_workers(self) -> int:
        return len(self._slots)

    def close(self) -> None:
        for _ in self._threads:
            self._tasks.put(None)
        for t in self._threads:
            t.join()
        for s in self._slots:
            s.close()

    def _loop(self, slot: _Slot) -> None:
        while True:
            item = self._tasks.get()
            if item is None:
                return
            jobs, fut = item
            try:
                fut.set_result([slot.run(j) for j in jobs])
            except BaseException as e:  # keep the dispatcher alive; surface to caller
                fut.set_exception(e)

    def _submit(self, jobs: list[dict]) -> Future:
        """Run `jobs` back to back on one worker."""
        fut: Future = Future()
        self._tasks.put((jobs, fut))
        return fut

    @staticmethod
    def _job(spec: dict, code: str, timeout_s: float) -> dict:
        # Expected outputs deliberately absent: workers never see them.
        return {"setup": spec["setup"], "code": code, "entry": spec["entry"],
                "tests": spec["tests"], "timeout_s": timeout_s}

    @staticmethod
    def _judge(run, expected: list) -> dict:
        if run is None:
            return {"verdict": INFRA_ERROR, "error": "worker_died", "cpu_s": None, "wall_s": None,
                    "num_tests": len(expected), "tests_passed": 0, "first_fail": None}
        return verifier.judge(run, expected)

    # --- gold timing ---

    def _ensure_gold(self, problems: dict, decoded: dict) -> None:
        """Measure gold for every spec not yet cached; all runs go out in parallel."""
        with self._gold_lock:
            todo = {h: problems[h] for h in problems if h not in self._gold}
        futs = {h: [self._submit([self._job(s, s["gold"], self.timeout_s)]) for _ in range(self.gold_runs)]
                for h, s in todo.items()}
        for h, fs in futs.items():
            results = [self._judge(f.result()[0], decoded[h]) for f in fs]
            bad = [r for r in results if r["verdict"] != PASS]
            entry = ({"t": None, "error": f"{bad[0]['verdict']}:{bad[0]['error']}"} if bad
                     else {"t": statistics.median(r["cpu_s"] for r in results), "error": None})
            with self._gold_lock:
                self._gold[h] = entry

    def measure_gold(self, problems: dict) -> dict:
        """{key: {"t": median gold CPU s or None, "error": None or "VERDICT:detail"}}."""
        decoded = {h: [json.loads(e) for e in s["expected"]] for h, s in problems.items()}
        self._ensure_gold(problems, decoded)
        return {h: self._gold[h] for h in problems}

    def _limit(self, t_gold) -> float | None:
        return None if t_gold is None else max(self.k * t_gold, self.floor_s)

    # --- public ---

    def verify(self, problems: dict, jobs: list) -> list[dict]:
        decoded = {h: [json.loads(e) for e in s["expected"]] for h, s in problems.items()}
        self._ensure_gold(problems, decoded)

        # A submission that runs past several limits fails either way; killing it
        # early keeps a step's slowest sample from stalling the batch.
        def timeout_for(h):
            lim = self._limit(self._gold[h]["t"])
            return self.timeout_s if lim is None else min(self.timeout_s, max(4 * lim, 1.0))

        futs = [None if code is None else self._submit([self._job(problems[h], code, timeout_for(h))])
                for h, code in jobs]
        results = []
        for (h, code), fut in zip(jobs, futs):
            gold = self._gold[h]
            meta = {"gold_s": gold["t"], "gold_error": gold["error"], "limit_s": self._limit(gold["t"]),
                    "rechecked": False}
            if fut is None:
                results.append({"verdict": MALFORMED, "error": None, "cpu_s": None, "wall_s": None,
                                "num_tests": len(decoded[h]), "tests_passed": 0, "first_fail": None, **meta})
                continue
            r = self._judge(fut.result()[0], decoded[h])
            if r["verdict"] == PASS and meta["limit_s"] is not None and r["cpu_s"] > meta["limit_s"]:
                r, meta = self._recheck(problems[h], code, decoded[h], timeout_for(h), meta)
            results.append({**r, **meta})
        return results

    def _recheck(self, spec, code, expected, timeout_s, meta):
        """Gold then submission on one worker; that pair alone sets the verdict."""
        g_run, s_run = self._submit([self._job(spec, spec["gold"], self.timeout_s),
                                     self._job(spec, code, timeout_s)]).result()
        g, r = self._judge(g_run, expected), self._judge(s_run, expected)
        limit = self._limit(g["cpu_s"]) if g["verdict"] == PASS else meta["limit_s"]
        meta = {**meta, "limit_s": limit, "rechecked": True}
        if r["verdict"] == PASS and r["cpu_s"] > limit:
            r = {**r, "verdict": TOO_SLOW}
        return r, meta

    def stats(self) -> dict:
        return {"workers": [s.cpu for s in self._slots], "restarts": sum(s.restarts for s in self._slots),
                "gold_cached": len(self._gold)}


# ---------------------------------------------------------------------------
# Socket server / client
# ---------------------------------------------------------------------------


def serve(pool: VerifierPool, address: str) -> None:
    if os.path.exists(address):
        os.unlink(address)
    old = os.umask(0o177)  # socket is user-only
    try:
        listener = Listener(address, family="AF_UNIX")
    finally:
        os.umask(old)

    def handle(conn):
        with conn:
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    return
                op = msg[0]
                if op == "verify":
                    conn.send(pool.verify(msg[1], msg[2]))
                elif op == "stats":
                    conn.send(pool.stats())
                elif op == "ping":
                    conn.send("pong")

    while True:
        threading.Thread(target=handle, args=(listener.accept(),), daemon=True).start()


class VerifierClient:
    """Thread-safe client. One connection per process, opened lazily."""

    def __init__(self, address: str | None = None):
        self.address = address or os.environ[SOCKET_ENV]
        self._conn = None
        self._pid = None
        self._lock = threading.Lock()

    def _call(self, *msg):
        with self._lock:
            if self._conn is None or self._pid != os.getpid():  # reconnect after fork
                self._conn = Client(self.address, family="AF_UNIX")
                self._pid = os.getpid()
            self._conn.send(msg)
            return self._conn.recv()

    def verify(self, problems: dict, jobs: list) -> list[dict]:
        return self._call("verify", problems, jobs)

    def stats(self) -> dict:
        return self._call("stats")


_default_client: VerifierClient | None = None


def default_client() -> VerifierClient:
    """Client for $VERIFIER_SOCKET; without it, start a private service.

    The private service runs on this process's physical cores and is stopped
    at exit. `train_rl` always sets the variable; this fallback is for
    `evaluate` and ad-hoc checks.
    """
    global _default_client
    if _default_client is None:
        if SOCKET_ENV not in os.environ:
            import atexit
            import tempfile
            address = os.path.join(tempfile.mkdtemp(prefix="verifier-"), "v.sock")
            proc = launch(physical_cores(), address)
            atexit.register(lambda: (proc.terminate(), proc.wait()))
            os.environ[SOCKET_ENV] = address
        _default_client = VerifierClient()
    return _default_client


def launch(cpus: list[int], address: str, startup_timeout_s: float = 120, **pool_kwargs) -> subprocess.Popen:
    """Start the service as a fresh interpreter and wait until it answers."""
    cmd = [sys.executable, "-m", "pipeline.tasks.leetcode.verifier", "--socket", address,
           "--cpus", ",".join(map(str, cpus))]
    for key, val in pool_kwargs.items():
        cmd += [f"--{key.replace('_', '-')}", str(val)]
    if os.path.exists(address):
        os.unlink(address)
    proc = subprocess.Popen(cmd)
    deadline = time.monotonic() + startup_timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"verifier service exited with code {proc.returncode}")
        try:
            conn = Client(address, family="AF_UNIX")
            conn.send(("ping",))
            conn.recv()
            conn.close()
            return proc
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.2)
    proc.kill()
    raise TimeoutError("verifier service did not start")


def main() -> None:
    p = argparse.ArgumentParser(description="LeetCode verifier service")
    p.add_argument("--socket", required=True)
    p.add_argument("--cpus", help="comma-separated CPU ids, used as given (default: one per physical core of own affinity)")
    p.add_argument("--memory-mb", type=int, default=2048)
    p.add_argument("--k", type=float, default=3.0)
    p.add_argument("--floor-s", type=float, default=0.05)
    p.add_argument("--gold-runs", type=int, default=3)
    p.add_argument("--timeout-s", type=float, default=10.0)
    args = p.parse_args()
    cpus = [int(c) for c in args.cpus.split(",")] if args.cpus else physical_cores()
    # The dispatcher shares the workers' cores; it mostly waits on pipes.
    os.sched_setaffinity(0, cpus)
    pool = VerifierPool(cpus, memory_mb=args.memory_mb, k=args.k, floor_s=args.floor_s,
                        gold_runs=args.gold_runs, timeout_s=args.timeout_s)
    print(f"verifier: {len(cpus)} workers on cpus {cpus}, socket {args.socket}", flush=True)
    serve(pool, args.socket)

