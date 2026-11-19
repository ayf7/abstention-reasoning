"""Parallel `codex exec` calls: at most `sessions` at once, one throwaway session per prompt.

A session keeps nothing: --ephemeral writes no session file, and its state
databases live in a temp directory that is deleted when it ends, so no SQLite
file on the NFS home is shared between sessions. Auth still comes from ~/.codex.
"""
import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

MODEL = "gpt-5.6-sol"
EFFORT = "high"
SESSIONS = 8
TIMEOUT = 1800  # seconds per session
TRIES = 3
FATAL = re.compile(r"usage limit|rate.?limit|quota|\b429\b|\b401\b|unauthorized", re.I)


class Fatal(RuntimeError):
    """The account cannot serve more requests now; retrying would only add load."""


def _errors(out: str) -> list[str]:
    """Error messages among `codex exec --json` events."""
    errs = []
    for line in out.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") == "error":
            errs.append(str(e.get("message")))
        elif e.get("type") == "turn.failed":
            errs.append(str(e.get("error")))
    return errs


def _tokens(out: str) -> int:
    """Tokens used, not counting cached input, from the last `turn.completed` event."""
    n = 0
    for line in out.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") == "turn.completed":
            u = e["usage"]
            n = u["input_tokens"] - u["cached_input_tokens"] + u["output_tokens"]
    return n


def ask(prompt: str, schema: dict) -> tuple[str, int]:
    """(final reply, tokens used) from one new session whose reply follows `schema`."""
    with tempfile.TemporaryDirectory(prefix="codex_session_") as d:
        d = Path(d)
        (d / "work").mkdir()
        (d / "db").mkdir()
        (d / "schema.json").write_text(json.dumps(schema))
        cmd = ["codex", "exec", "--json", "--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
               "--sandbox", "read-only", "-C", str(d / "work"), "-m", MODEL,
               "-c", f"model_reasoning_effort={EFFORT}", "-c", "project_doc_max_bytes=0", "-c", "web_search=disabled",
               "-c", f'sqlite_home="{d / "db"}"', "--output-schema", str(d / "schema.json"),
               "-o", str(d / "reply"), "-"]  # prompt on stdin: argv caps one argument at 128 KB
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, start_new_session=True)
        try:
            out, _ = proc.communicate(prompt, timeout=TIMEOUT)
        finally:
            if proc.poll() is None:  # codex is a wrapper around a child, so end the whole group
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        errs = _errors(out)
        if any(FATAL.search(e) for e in errs):
            raise Fatal("; ".join(errs)[:500])
        if proc.returncode or not (d / "reply").exists():
            raise RuntimeError(f"exit {proc.returncode}: {'; '.join(errs)[:500]}")
        return (d / "reply").read_text(), _tokens(out)


def ask_all(prompts: dict[str, str], schema: dict, cache: Path, check: Callable[[str, dict], None],
            sessions: int = SESSIONS) -> dict[str, str]:
    """{key: reply} for the prompts that got one.

    `check(key, parsed reply)` raises ValueError on a reply to retry. Replies are
    cached in `cache` by key, so an interrupted run resumes. A Fatal
    error stops new sessions from starting; sessions already running finish.
    """
    cache.mkdir(parents=True, exist_ok=True)
    stop, lock = threading.Event(), threading.Lock()
    done, tokens = [0], [0]

    def one(key: str) -> None:
        f = cache / f"{key}.json"
        if f.exists() or stop.is_set():
            return
        for attempt in range(TRIES):
            if stop.is_set():
                return
            try:
                reply, used = ask(prompts[key], schema)
                check(key, json.loads(reply))
                break
            except Fatal as e:
                stop.set()
                print(f"stopping, no new sessions: {e}", flush=True)
                return
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as e:
                print(f"  {key} attempt {attempt + 1}: {e!r}"[:500], flush=True)
                if attempt == TRIES - 1:
                    return
                time.sleep(30 * (attempt + 1))
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"reply": reply, "tokens": used}))
        tmp.rename(f)
        with lock:
            done[0] += 1
            tokens[0] += used
            print(f"{done[0]} new replies, {tokens[0]:,} tokens", flush=True)

    with ThreadPoolExecutor(sessions) as ex:
        list(ex.map(one, prompts))
    return {k: json.loads((cache / f"{k}.json").read_text())["reply"] for k in prompts if (cache / f"{k}.json").exists()}
