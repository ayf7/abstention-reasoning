"""Run untrusted Python in a confined, forked child and measure its CPU time.

This module is imported by verifier workers. A worker builds a `Sandbox` once
(Landlock ruleset, seccomp filter) and then calls `Sandbox.run(job)` per
submission. Each run forks a child that confines itself, executes the code on
the job's test calls, and streams one JSON line per test back over a pipe. The
child never receives expected outputs: comparison happens in the dispatcher,
so nothing the submission can inspect reveals the answers.

Confinement (no root, no namespaces on this cluster):
  - every inherited fd closed; stdio -> /dev/null; new session
  - rlimits: address space, CPU seconds, file size 0, open files, no core
  - Landlock: read-only access to the Python installation and system libs
  - seccomp: EPERM for exec, fork/clone (threads too), sockets, ptrace, kill,
    namespace/mount calls
The worker enforces a wall-clock deadline (killpg + SIGKILL) and caps output.
CPU time comes from wait4's rusage, which the child cannot forge.
"""

import ctypes
import json
import os
import resource
import select
import signal
import sys
import sysconfig
import time

# Exit code the child uses when confinement itself fails (before any
# untrusted code runs). Paired with the absence of the READY line, it marks an
# infrastructure error rather than a submission failure.
CONFINE_FAILED_EXIT = 120
READY = b'{"ready":1}\n'

# --- Landlock (x86_64 syscall numbers; ABI >= 3 has bits 0..14) ---
_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_RULE_PATH_BENEATH = 1
_FS_EXECUTE = 1 << 0
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_HANDLED_ABI3 = (1 << 15) - 1
_PR_SET_NO_NEW_PRIVS = 38
# os.pidfd_open is missing from some Python builds; call the syscall directly.
_SYS_PIDFD_OPEN = 434

# --- seccomp (libseccomp via ctypes) ---
_SCMP_ACT_ALLOW = 0x7FFF0000
_SCMP_ACT_ERRNO_EPERM = 0x00050000 | 1
_DENIED_SYSCALLS = [
    "execve", "execveat", "fork", "vfork", "clone", "clone3",
    "socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
    "ptrace", "process_vm_readv", "process_vm_writev",
    "kill", "tkill", "tgkill", "rt_sigqueueinfo", "rt_tgsigqueueinfo",
    "pidfd_open", "pidfd_send_signal", "pidfd_getfd",
    "setsid", "setpgid", "unshare", "setns", "mount", "umount2", "pivot_root",
    "chroot", "keyctl", "add_key", "request_key", "bpf", "perf_event_open",
    "userfaultfd", "io_uring_setup", "io_uring_enter", "io_uring_register",
    "personality", "mknod", "mknodat",
]


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _readable_roots() -> list[str]:
    """Directories the child may read: the interpreter's install and system libs.

    Deliberately excludes sys.path entries such as the repo root, which holds
    the dataset and its expected outputs.
    """
    paths = sysconfig.get_paths()
    roots = {sys.base_prefix, sys.prefix, sys.exec_prefix,
             paths["stdlib"], paths["platstdlib"], paths["purelib"], paths["platlib"],
             "/usr/lib", "/lib", "/lib64", "/usr/lib64"}
    return sorted(r for r in roots if os.path.isdir(r))


def _landlock_ruleset(libc) -> int:
    attr = ctypes.c_uint64(_FS_HANDLED_ABI3)
    fd = libc.syscall(_SYS_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.c_size_t(8), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    for root in _readable_roots():
        pfd = os.open(root, os.O_PATH | os.O_CLOEXEC)
        rule = _PathBeneath(_FS_READ_FILE | _FS_READ_DIR | _FS_EXECUTE, pfd)
        rc = libc.syscall(_SYS_LANDLOCK_ADD_RULE, fd, _LANDLOCK_RULE_PATH_BENEATH, ctypes.byref(rule), 0)
        os.close(pfd)
        if rc != 0:
            raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {root}")
    return fd


def _seccomp_filter():
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    ctx = lib.seccomp_init(_SCMP_ACT_ALLOW)
    if not ctx:
        raise OSError("seccomp_init failed")
    for name in _DENIED_SYSCALLS:
        nr = lib.seccomp_syscall_resolve_name(name.encode())
        if nr < 0:  # not present on this architecture
            continue
        if lib.seccomp_rule_add(ctx, _SCMP_ACT_ERRNO_EPERM, nr, 0) != 0:
            raise OSError(f"seccomp_rule_add failed for {name}")
    return lib, ctx


# ---------------------------------------------------------------------------
# Output encoding (child side). The dispatcher's `outputs.match` reads this.
# ---------------------------------------------------------------------------

_MAX_NODES = 1_000_000


def encode(x):
    """Canonical JSON-able form of a return value.

    Exact type checks (not isinstance) so subclasses with custom __eq__ cannot
    masquerade as builtins. Linked lists and trees are recognised by shape
    (val/next, val/left/right) since submissions may define their own classes.
    """
    t = type(x)
    if x is None or t is bool or t is int or t is float or t is str:
        return x
    if t is list or t is tuple:
        return [encode(v) for v in x]
    if t is dict:
        return {"__dict__": [[encode(k), encode(v)] for k, v in x.items()]}
    if t is set or t is frozenset:
        return {"__set__": [encode(v) for v in x]}
    if hasattr(x, "val") and hasattr(x, "next"):
        vals, node = [], x
        while node is not None and len(vals) < _MAX_NODES:
            vals.append(encode(node.val))
            node = node.next
        return {"__list__": vals, "cyclic": node is not None}
    if hasattr(x, "val") and hasattr(x, "left") and hasattr(x, "right"):
        out, queue, i = [], [x], 0
        while i < len(queue) and len(queue) < _MAX_NODES:
            node = queue[i]
            i += 1
            if node is None:
                out.append(None)
                continue
            out.append(encode(node.val))
            queue.append(node.left)
            queue.append(node.right)
        while out and out[-1] is None:
            out.pop()
        return {"__tree__": out}
    return {"__type__": t.__name__}


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


class Sandbox:
    """Per-worker confinement state, built once and reused by every fork."""

    def __init__(self, memory_mb: int = 2048, max_output_bytes: int = 32 << 20):
        self.memory_mb = memory_mb
        self.max_output_bytes = max_output_bytes
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._ruleset_fd = _landlock_ruleset(self._libc)
        self._seccomp_lib, self._seccomp_ctx = _seccomp_filter()
        self._devnull = os.open(os.devnull, os.O_RDWR)

    # --- child side ---

    def _confine(self, wfd: int, cpu_seconds: int) -> None:
        for fd in (0, 1, 2):
            os.dup2(self._devnull, fd)
        # Close only the fds that are open: closerange over the whole fd table
        # costs ~7 ms of child CPU, which would land in the measured time.
        keep = {0, 1, 2, wfd, self._ruleset_fd}
        for name in os.listdir("/proc/self/fd"):
            fd = int(name)
            if fd not in keep:
                try:
                    os.close(fd)
                except OSError:  # the fd listdir itself used
                    pass
        os.setsid()

        mem = self.memory_mb << 20
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        stack = mem // 4
        resource.setrlimit(resource.RLIMIT_STACK, (stack, stack))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        if self._libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(NO_NEW_PRIVS) failed")
        if self._libc.syscall(_SYS_LANDLOCK_RESTRICT_SELF, self._ruleset_fd, 0) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
        os.close(self._ruleset_fd)
        if self._seccomp_lib.seccomp_load(self._seccomp_ctx) != 0:
            raise OSError("seccomp_load failed")

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            view = view[n:]

    def _child(self, job: dict, wfd: int) -> None:
        try:
            self._confine(wfd, job["cpu_seconds"])
            self._write_all(wfd, READY)
        except BaseException:
            os._exit(CONFINE_FAILED_EXIT)

        # Untrusted code runs below. Anything it does only changes the data it
        # writes back; the verdict is decided elsewhere.
        write, dumps, _exit = self._write_all, json.dumps, os._exit
        try:
            sys.setrecursionlimit(100_000)
            ns = {"__name__": "solution", "__builtins__": __builtins__}
            exec(compile(job["setup"], "<setup>", "exec"), ns)
            calls = [compile(f"({job['entry']})({args})", "<test>", "eval") for args in job["tests"]]
            exec(compile(job["code"], "<solution>", "exec"), ns)
        except BaseException as e:
            write(wfd, dumps({"error": type(e).__name__, "stage": "load"}).encode() + b"\n")
            _exit(0)

        for call in calls:
            try:
                line = dumps({"out": encode(eval(call, ns))})
            except BaseException as e:
                write(wfd, dumps({"error": type(e).__name__}).encode() + b"\n")
                _exit(0)
            write(wfd, line.encode() + b"\n")
        _exit(0)

    # --- worker side ---

    def run(self, job: dict) -> dict:
        """Fork, confine, run, and collect. Never raises for submission behaviour.

        job: {"setup", "code", "entry", "tests": [args_src], "timeout_s"}
        """
        timeout_s = float(job["timeout_s"])
        job = {**job, "cpu_seconds": int(timeout_s) + 1}
        rfd, wfd = os.pipe()
        start = time.perf_counter()
        pid = os.fork()
        if pid == 0:  # child
            try:
                os.close(rfd)
                self._child(job, wfd)
            finally:
                os._exit(CONFINE_FAILED_EXIT)
        os.close(wfd)

        pidfd = self._libc.syscall(_SYS_PIDFD_OPEN, pid, 0)
        if pidfd < 0:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            raise OSError(ctypes.get_errno(), "pidfd_open failed")
        poller = select.poll()
        poller.register(rfd, select.POLLIN)
        poller.register(pidfd, select.POLLIN)
        chunks, size, outcome = [], 0, None
        deadline = start + timeout_s
        pipe_open, exited = True, False
        while pipe_open or not exited:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                outcome = "timeout"
                break
            for fd, _ in poller.poll(remaining * 1000):
                if fd == rfd:
                    data = os.read(rfd, 1 << 16)
                    if not data:
                        pipe_open = False
                        poller.unregister(rfd)
                        continue
                    chunks.append(data)
                    size += len(data)
                    if size > self.max_output_bytes:
                        outcome = "output_limit"
                elif fd == pidfd:
                    exited = True
                    poller.unregister(pidfd)
            if outcome:
                break
        if outcome:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _, status, ru = os.wait4(pid, 0)
        wall = time.perf_counter() - start
        os.close(rfd)
        os.close(pidfd)
        return {
            "outcome": outcome or "exited",
            "exit_code": os.waitstatus_to_exitcode(status),
            "stdout": b"".join(chunks),
            "cpu_s": ru.ru_utime + ru.ru_stime,
            "wall_s": wall,
            "maxrss_kb": ru.ru_maxrss,
        }
