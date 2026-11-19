"""Compare a program's runtime with gold's on AI-written inputs (see leetcode/runtime_tests.py).

Series time only the solution's call, in CPU time from inside the process, taking
the fastest of several repeats; a program fails if its cost grows faster than gold's
or it runs out of time where gold finished. Limit cases run once: a program fails
if it takes far longer than gold. Timings depend on the machine, so measure gold and
the program on the same one. Everything runs confined from outside, with the same
Landlock and seccomp rules as sandbox.py, except that exec stays possible for the
one binary started.
"""
import ctypes
import inspect
import json
import math
import os
import re
import resource
import select
import shutil
import signal
import sys
import tempfile
import time

from . import sandbox as sbx

PYTHON = os.path.realpath(sys.executable)
LOADER = os.path.realpath("/lib64/ld-linux-x86-64.so.2")
TMP = os.environ.get("RUNTIME_TMP", tempfile.gettempdir())

SERIES_TIMEOUT = 30  # seconds per run, repeats included, on top of what the whole run took gold (building a huge input is not the program's fault)
REPEATS, REPEATS_MIN, REPEATS_NS, REPEATS_WALL = 9, 3, 10**9, 10  # up to this many calls per run, stopping after the minimum once this much CPU is spent or this many seconds have passed (building a huge input can outweigh a fast call)
MARGIN = 0.25  # a slope this far above gold's fails, if the program is also MIN_RATIO times gold's cost at its largest size
MIN_RATIO = 3.0
STEADY_RATIO, STEADY_NS = 10.0, 100_000_000  # this many times gold's cost fails whatever the slope, once the program costs this much at its largest size
WINDOW = 3  # slopes span the largest this many sizes a program finished
MIN_NS = 2_000  # below this a call is interpreter noise, not the program's growth
COLLAPSE, COLLAPSE_FROM = 0.5, 100_000  # gold under this share of its cost at a smaller size took a shortcut, once that cost is this many ns
HOLLOW_LIMIT = 0.1  # gold under this share of its slowest series point on a limit case took a shortcut there
LIMIT_RATIO, LIMIT_FLOOR, LIMIT_CAP = 10.0, 1.0, 120.0  # limit-case budget in CPU seconds: max(ratio * gold, floor), up to cap

_DRIVER = r'''
import copy, ctypes, functools, gc, hashlib, json, operator, random, sys, time
job = json.load(sys.stdin)
sys.setrecursionlimit(10_000_000)  # the memo below adds a frame per level, so depth 1,000,000 takes 2,000,000
ctypes.pythonapi.PyThreadState_Get.restype = ctypes.c_void_p
tstate = (ctypes.c_int * 16).from_address(ctypes.pythonapi.PyThreadState_Get())  # py_recursion_limit, then c_recursion_remaining, in CPython 3.12
assert tstate[8] == 10_000_000 and 9000 < tstate[9] <= 10000, "not the CPython 3.12 thread state"
tstate[9] = 2_000_000_000  # recursion through C, as in max(dfs(x) for x in g), otherwise fails near depth 4000 whatever the limit
def memo(fn):  # the C cache wrappers raise RecursionError between depth 5000 and 9000; this one does not
    table = {}
    def wrapper(*a, **k):
        key = (a, tuple(k.items())) if k else a
        if key in table:
            return table[key]
        r = table[key] = fn(*a, **k)
        return r
    wrapper.cache_clear = table.clear
    return functools.update_wrapper(wrapper, fn)
lru_cache = functools.lru_cache
functools.cache = memo
functools.lru_cache = lambda maxsize=128, typed=False: memo if maxsize is None else lru_cache(maxsize, typed)
exec(job["encode"])
gen = {"__builtins__": __builtins__}
exec(compile(job["generator"], "<generator>", "exec"), gen)
raw = gen["make"](random.Random(job["seed"]), job["name"], job["size"])
if job["check"]:
    print("LEGAL", gen["legal"](raw) is True)
    print("BYTES", len(json.dumps(raw)))
clock, call = time.thread_time_ns, operator.call  # under RLIMIT_CPU process_time only moves in 1 ms ticks; seccomp forbids other threads
best, spent, start = None, 0, time.time()
for rep in range(job["repeats"]):
    harness = {"__builtins__": __builtins__}
    exec(compile(job["setup"], "<setup>", "exec"), harness)
    ns = {"__name__": "solution", "__builtins__": __builtins__}  # new each time, so no call inherits another's memo
    exec(compile(job["code"], "<solution>", "exec"), ns)
    scope = {**ns, **harness}
    if "build" in gen:
        args = gen["build"](copy.deepcopy(raw), ns, harness)
    else:
        args = {k: scope[job["nodes"][k]](x) if k in job["nodes"] else x for k, x in copy.deepcopy(raw).items()}
    fn = eval(job["entry"], scope)
    gc.collect()
    t0 = clock()
    out = call(fn, **args)
    t1 = clock()
    best, spent = t1 - t0 if best is None else min(best, t1 - t0), spent + t1 - t0
    if rep + 1 >= job["repeats_min"] and (spent > job["repeats_ns"] or time.time() - start > job["repeats_wall"]):
        break
if best is not None:
    print("RES", hashlib.md5(json.dumps(encode(out)).encode()).hexdigest())
    print("NS", best)
print("ok")
'''
_ENCODE = f"_MAX_NODES = {sbx._MAX_NODES}\n" + inspect.getsource(sbx.encode)
_libc = ctypes.CDLL(None, use_errno=True)
_READ, _WRITE = sbx._FS_READ_FILE | sbx._FS_READ_DIR, (1 << 1) | (1 << 5) | (1 << 8)  # WRITE_FILE, REMOVE_FILE, MAKE_REG


def _rule(fd: int, path: str, access: int) -> None:
    p = os.open(path, os.O_PATH | os.O_CLOEXEC)
    rc = _libc.syscall(sbx._SYS_LANDLOCK_ADD_RULE, fd, sbx._LANDLOCK_RULE_PATH_BENEATH, ctypes.byref(sbx._PathBeneath(access, p)), 0)
    os.close(p)
    if rc:
        raise OSError(ctypes.get_errno(), f"landlock rule failed: {path}")


def _confine(tmp: str) -> None:
    attr = ctypes.c_uint64(sbx._FS_HANDLED_ABI3)
    fd = _libc.syscall(sbx._SYS_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.c_size_t(8), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    for root in sbx._readable_roots():
        _rule(fd, root, _READ)
    for path in (PYTHON, LOADER):
        _rule(fd, path, sbx._FS_READ_FILE | sbx._FS_EXECUTE)
    _rule(fd, "/dev/null", sbx._FS_READ_FILE | (1 << 1))
    _rule(fd, tmp, _READ | _WRITE | (1 << 1))
    for limit, value in ((resource.RLIMIT_AS, 16 << 30), (resource.RLIMIT_STACK, 512 << 20), (resource.RLIMIT_CPU, 900),
                         (resource.RLIMIT_FSIZE, 1 << 16), (resource.RLIMIT_CORE, 0), (resource.RLIMIT_NOFILE, 64)):
        resource.setrlimit(limit, (value, value))
    if _libc.prctl(sbx._PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) or _libc.syscall(sbx._SYS_LANDLOCK_RESTRICT_SELF, fd, 0):
        raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    os.close(fd)
    saved, sbx._DENIED_SYSCALLS = sbx._DENIED_SYSCALLS, [n for n in sbx._DENIED_SYSCALLS if n not in ("execve", "execveat")]
    lib, ctx = sbx._seccomp_filter()
    sbx._DENIED_SYSCALLS = saved
    if lib.seccomp_load(ctx):
        raise OSError("seccomp_load failed")


def run(spec: dict, code: str, generator: str, name: str, size: int | None, *, repeats: int = 1, check: bool = False,
        seed: int = 0, timeout: float = SERIES_TIMEOUT) -> dict:
    """One confined run of `code` on make(rng, name, size).

    spec: setup, entry, nodes (argument name -> builder). Returns ok, timed, res
    (hash of the result), err, secs, ns (the fastest call's CPU); legal and bytes
    when `check`. repeats=0 does everything except the call.
    """
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="rt", dir=TMP)
    job = dict(setup=spec["setup"], entry=spec["entry"], nodes=spec["nodes"], code=code, generator=generator, name=name, size=size,
               seed=seed, check=check, repeats=repeats, repeats_min=REPEATS_MIN, repeats_ns=REPEATS_NS, repeats_wall=REPEATS_WALL, encode=_ENCODE)
    mfd = _libc.memfd_create(b"job", 0)
    os.write(mfd, json.dumps(job).encode())
    os.lseek(mfd, 0, 0)
    out_r, out_w = os.pipe()
    argv = [PYTHON, "-s", "-P", "-c", _DRIVER]
    env = {"PYTHONHASHSEED": "0", "TMPDIR": tmp, "PATH": "/usr/bin"}
    pid = os.fork()
    if pid == 0:
        try:
            os.setsid()
            os.dup2(mfd, 0)
            os.dup2(out_w, 1)
            os.dup2(out_w, 2)
            os.closerange(3, 1 << 16)
            _confine(tmp)
            os.execve(PYTHON, argv, env)
        except BaseException as e:
            os.write(1, f"confine failed: {e!r}\n".encode())
        finally:
            os._exit(sbx.CONFINE_FAILED_EXIT)
    os.close(out_w)
    os.close(mfd)
    buf, timed, deadline = b"", False, t0 + timeout
    while True:
        left = deadline - time.time()
        if left <= 0:
            timed = True
            break
        if select.select([out_r], [], [], left)[0]:
            chunk = os.read(out_r, 1 << 16)
            if not chunk:
                break
            buf += chunk[:1 << 24]
    if timed:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _, status = os.waitpid(pid, 0)
    os.close(out_r)
    shutil.rmtree(tmp, ignore_errors=True)
    out = buf.decode(errors="replace")
    tag = lambda t: next((x[len(t) + 1:] for x in out.splitlines() if x.startswith(t + " ")), None)
    ok = not timed and status == 0 and "ok" in out.splitlines()
    r = dict(ok=ok, timed=timed, res=tag("RES"), secs=round(time.time() - t0, 2), err=None if ok or timed else out[-300:])
    if check:
        r |= dict(legal=tag("LEGAL") == "True", bytes=int(tag("BYTES") or 0))
    r["ns"] = int(tag("NS")) if ok and tag("NS") else None
    return r


def series_points(spec: dict, code: str, generator: str, series: dict, sizes: list[int] | None = None, check: bool = False,
                  timeout: float = SERIES_TIMEOUT) -> list[dict]:
    """Timed runs at ascending sizes, stopping at the first failure."""
    points = []
    for size in sizes if sizes is not None else series["sizes"]:
        points.append({"size": size, **run(spec, code, generator, series["name"], size, repeats=REPEATS, check=check, timeout=timeout)})
        if not points[-1]["ok"]:
            break
    return points


def limit_cpu(spec: dict, code: str, generator: str, name: str, timeout: float, check: bool = False) -> dict:
    """CPU seconds of the call on a limit case; `timeout` is the budget for the call alone, not for building the input."""
    base = run(spec, code, generator, name, None, repeats=0, check=check, timeout=LIMIT_CAP)
    if not base["ok"]:
        return {**base, "net": None}
    r = run(spec, code, generator, name, None, timeout=timeout + 10 + 4 * base["secs"])  # the verdict comes from CPU; wall time varies with load, so this only ends runs far over
    return {**r, "legal": base.get("legal"), "bytes": base.get("bytes"), "net": r["ns"] / 1e9 if r["ok"] else None}


def slope(points: list[dict]) -> float | None:
    """Growth exponent of CPU with size across the last WINDOW usable points."""
    pts = [p for p in points if p.get("ns") and p["ns"] >= MIN_NS and p["size"] > 0][-WINDOW:]
    if len(pts) < 2 or pts[0]["size"] == pts[-1]["size"]:
        return None
    return math.log(pts[-1]["ns"] / pts[0]["ns"]) / math.log(pts[-1]["size"] / pts[0]["size"])


def score_series(program: list[dict], gold: list[dict]) -> dict:
    """gap: program's slope minus gold's on the sizes the program finished; ratio: program's cost over gold's at the largest.

    slower: a timeout or crash where gold finished, or both gap >= MARGIN and ratio >= MIN_RATIO. The sizes reach the constraint's
    limit, so a steeper slope that stays within a small factor of gold there cannot hurt on any allowed input. A large steady
    factor can, so STEADY_RATIO times gold also fails once the program takes STEADY_NS.
    """
    by_size = {p["size"]: p for p in gold if p["ok"]}
    mine = [p for p in program if p["ok"] and p["size"] in by_size]
    a, b = slope(mine), slope([by_size[p["size"]] for p in mine])
    gap = None if a is None else a - (b if b is not None else 0.0)  # gold too small to measure grows like a constant
    ratio = mine[-1]["ns"] / max(by_size[mine[-1]["size"]]["ns"], MIN_NS) if mine else None
    timed = bool(program) and program[-1]["timed"] and program[-1]["size"] in by_size
    crashed = bool(program) and not program[-1]["ok"] and not program[-1]["timed"] and program[-1]["size"] in by_size
    steady = ratio is not None and ratio >= STEADY_RATIO and mine[-1]["ns"] >= STEADY_NS
    return dict(gap=gap, ratio=ratio, timed=timed, crashed=crashed, slower=timed or crashed or steady or (gap is not None and gap >= MARGIN and ratio >= MIN_RATIO),
                differs=sum(p["res"] != by_size[p["size"]]["res"] for p in mine))


def full_work(points: list[dict]) -> list[dict]:
    """Gold's points without the sizes where it collapsed, such as a maximum length it rejects before doing any work."""
    kept, peak = [], 0
    for p in points:
        if peak < COLLAPSE_FROM or p["ns"] >= COLLAPSE * peak:
            kept.append(p)
        peak = max(peak, p["ns"])
    return kept


def hollow_limit(net: float, series_ns: list[int]) -> bool:
    """Whether a limit case cost gold (net seconds) far less than its slowest series point (ns), so it never ran gold in full."""
    peak = max(series_ns, default=0)
    return peak >= COLLAPSE_FROM and net * 1e9 < HOLLOW_LIMIT * peak


def limit_budget(gold_net: float) -> float:
    return min(max(LIMIT_RATIO * gold_net, LIMIT_FLOOR), LIMIT_CAP)


def evaluate(spec: dict, code: str, tests: dict, gold: dict) -> dict:
    """Runtime verdict for `code`. tests: series, limit_cases, generator; gold: from measure_gold."""
    series = {}
    for s in tests["series"]:
        finished = full_work([p for p in gold["series"][s["name"]] if p["ok"]])
        points = series_points(spec, code, tests["generator"], s, [p["size"] for p in finished], timeout=SERIES_TIMEOUT + max(p["secs"] for p in finished)) if finished else None
        series[s["name"]] = points and score_series(points, gold["series"][s["name"]])
    limits = {}
    for c in tests["limit_cases"]:
        g = gold["limit"][c["name"]]
        if g["net"] is None:
            limits[c["name"]] = None
            continue
        budget = limit_budget(g["net"])
        r = limit_cpu(spec, code, tests["generator"], c["name"], budget)
        limits[c["name"]] = dict(net=r["net"], budget=budget, slower=not r["ok"] or r["net"] > budget,
                                 differs=r["ok"] and r["res"] != g["res"], err=r["err"])
    return dict(slower=any(x and x["slower"] for x in [*series.values(), *limits.values()]), series=series, limit=limits)


def measure_gold(spec: dict, tests: dict) -> dict:
    """Gold's points on every test, with its inputs checked by the generator's legal()."""
    return dict(series={s["name"]: series_points(spec, tests["gold"], tests["generator"], s, check=True) for s in tests["series"]},
                limit={c["name"]: limit_cpu(spec, tests["gold"], tests["generator"], c["name"], LIMIT_CAP, check=True) for c in tests["limit_cases"]})
