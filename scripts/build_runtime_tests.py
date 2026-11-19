"""Build data/leetcode_o2/runtime_tests.json: AI-written runtime tests for every problem, with gold's measurements.

    python scripts/build_runtime_tests.py                  # everything, then write the file
    python scripts/build_runtime_tests.py --only 2481,2530 --out x.json # just these problems
    python scripts/build_runtime_tests.py --revise notes.json  # also send these problems back, {index: what to change}

Codex writes each problem's tests. Each reply is then run against gold, and what
failed goes back to Codex, up to REPAIRS times. Gold is then timed. Replies, checks and measurements are cached, so a rerun resumes.
Problems whose tests still fail keep their closest reply and list `failures`.
"""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor

from pipeline.core import codex
from pipeline.core.io import save_json
from pipeline.core.method import DATA_ROOT
from pipeline.tasks.leetcode import limit_tests, runtime_tests as rt
from pipeline.tasks.leetcode.verifier import runtime

SRC = DATA_ROOT / "leetcode_o2" / "problems" / "primitives.json"
DST = DATA_ROOT / "leetcode_o2" / "runtime_tests.json"
CHECKS, GOLD = DST.with_name("runtime_tests_checks.jsonl"), DST.with_name("runtime_tests_gold.jsonl")
REPLIES = DATA_ROOT / ".codex_replies"
REPAIRS = 2
SERIES_CPU = 3.0  # CPU seconds of gold at a series' last size, leaving a slower program room within SERIES_TIMEOUT

REPAIR = """{prompt}

## Your previous reply
```json
{reply}
```

## What happened when we ran it
{failures}

Return the whole corrected reply. Change what these failures require and keep what works. If a failure cannot be fixed, such as the reference itself crashing on every input at the limit, keep the largest input it can handle and explain in `limitations`."""


def spec_of(p: dict) -> dict:
    return dict(setup=p["setup"], entry=p["entry"], nodes=limit_tests.node_args(p["starter_code"]))


def _fingerprint(key: str, reply: dict) -> str:
    return key + ":" + hashlib.sha1(json.dumps(reply, sort_keys=True).encode()).hexdigest()


def _problem(what: str, r: dict) -> str | None:
    """Why one run of gold is unusable, if it is."""
    if r["timed"]:
        return f"{what}: the reference did not finish"
    if not r["ok"]:
        return f"{what}: the generator or reference crashed: {r['err']}"
    if not r["legal"]:
        return f"{what}: your legal() returned False"
    return None


def check(job: tuple[dict, dict]) -> list[str]:
    """What is wrong with a reply, found by running gold on every input."""
    p, reply = job
    spec, failures, nets, slowest = spec_of(p), [], [], []
    for s in reply["series"]:
        costs, points = [], []
        for size in s["sizes"]:
            r = runtime.run(spec, p["gold"], reply["generator"], s["name"], size, repeats=runtime.REPEATS, check=True, timeout=10 * SERIES_CPU)
            if why := _problem(f"series {s['name']} at size {size}", r):
                failures.append(why)
                break
            costs.append((size, r["ns"] / 1e9))
            points.append({"size": size, "ns": r["ns"]})
            slowest.append(r["ns"])
        else:
            size, cpu = costs[-1]
            if cpu > SERIES_CPU:
                failures.append(f"series {s['name']} at size {size}: the reference took {cpu:.1f} CPU seconds; keep it near 2 by shrinking the other quantities")
            limit = rt.stated_limit(p, s)
            if limit is not None and size < rt.MIN_REACH * limit:
                r = runtime.run(spec, p["gold"], reply["generator"], s["name"], limit, check=True, timeout=10 * SERIES_CPU)
                if r["ok"] and r["legal"] and r["ns"] / 1e9 <= SERIES_CPU:
                    failures.append(f"series {s['name']} stops at size {size}, but size {limit} is legal and the reference handled it in {r['ns'] / 1e9:.2f} CPU seconds; extend it, and remove any cap in make")
            kept = {q["size"] for q in runtime.full_work(points)}
            for q in points:
                if q["size"] not in kept:
                    failures.append(f"series {s['name']} at size {q['size']}: the reference took {q['ns'] / 1e9:.6f} CPU seconds, far less than at a smaller size, so it took a shortcut; use the nearest input where it does its full work")
    for c in reply["limit_cases"]:
        r = runtime.limit_cpu(spec, p["gold"], reply["generator"], c["name"], runtime.LIMIT_CAP, check=True)
        if why := _problem(f"limit case {c['name']}", r):
            failures.append(why)
        else:
            nets.append(r["net"])
    if nets and runtime.hollow_limit(max(nets), slowest):
        failures.append(f"no limit case makes the reference do its full work: the slowest took {max(nets):.6f} CPU seconds, against {max(slowest) / 1e9:.6f} on a series; put every quantity at its limit without letting the reference take a shortcut")
    return failures


def run_cached(fn, jobs: dict[str, tuple], path, workers: int) -> dict[str, object]:
    """{fingerprint: fn(job)}, appending new results to the jsonl at `path`."""
    done = {}
    if path.exists():
        done = {d["id"]: d["out"] for d in map(json.loads, open(path))}
    todo = {k: j for k, j in jobs.items() if k not in done}
    print(f"{path.name}: {len(jobs) - len(todo)} cached, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(workers) as ex, open(path, "a") as out:
        for n, (k, r) in enumerate(zip(todo, ex.map(fn, todo.values())), 1):
            done[k] = r
            out.write(json.dumps(dict(id=k, out=r)) + "\n")
            out.flush()
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    return {k: done[k] for k in jobs}


def measure(job: tuple[dict, dict]) -> dict:
    p, reply = job
    return runtime.measure_gold(spec_of(p), {**reply, "gold": p["gold"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated problem indices")
    ap.add_argument("--out", default=DST)
    ap.add_argument("--revise", help="json file of {index: what to change}, sent back after the repairs")
    ap.add_argument("--stage", default="revise", help="reply cache for --revise; a new name asks Codex again")
    ap.add_argument("--sessions", type=int, default=codex.SESSIONS)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = {str(p["index"]): p for p in json.load(open(SRC))}
    if args.only:
        rows = {k: rows[k] for k in args.only.split(",")}
    prompts = {k: rt.format_prompt(p) for k, p in rows.items()}
    verify = lambda k, r: rt.validate(rows[k], r)

    best = {k: json.loads(r) for k, r in codex.ask_all(prompts, rt.SCHEMA, REPLIES / f"runtime_tests__{codex.MODEL}", verify, args.sessions).items()}
    if missing := sorted(set(rows) - set(best), key=int):
        raise SystemExit(f"{len(missing)} problems have no reply; rerun to retry: {missing[:20]}")

    def failures_of(replies: dict[str, dict]) -> dict[str, list[str]]:
        out = run_cached(check, {_fingerprint(k, r): (rows[k], r) for k, r in replies.items()}, CHECKS, args.workers)
        return {k: out[_fingerprint(k, r)] for k, r in replies.items()}

    def ask(stage: str, notes: dict[str, str], replies: dict[str, dict]) -> dict[str, dict]:
        asks = {k: REPAIR.format(prompt=prompts[k], reply=json.dumps(replies[k], indent=1), failures=n) for k, n in notes.items()}
        return {k: json.loads(r) for k, r in codex.ask_all(asks, rt.SCHEMA, REPLIES / f"runtime_tests_{stage}__{codex.MODEL}", verify, args.sessions).items()}

    def repair(replies: dict[str, dict], failures: dict[str, list[str]], stage: str = "") -> None:
        for attempt in range(1, REPAIRS + 1):
            bad = [k for k, f in failures.items() if f]
            print(f"after {stage}attempt {attempt - 1}: {len(bad)} of {len(replies)} problems have failures", flush=True)
            if not bad:
                break
            fixed = ask(f"{stage}repair{attempt}", {k: "\n".join(f"- {x}" for x in failures[k]) for k in bad}, replies)
            for k, f in failures_of(fixed).items():
                if len(f) <= len(failures[k]):
                    replies[k], failures[k] = fixed[k], f

    if args.revise:  # revise what is shipped, so an earlier revision is neither redone nor lost
        shipped = json.load(open(DST))
        best = {k: {f: shipped[k][f] for f in rt.SCHEMA["required"]} for k in rows}
        failures = {k: shipped[k]["failures"] for k in rows}
    else:
        failures = failures_of(best)
        repair(best, failures)

    if args.revise:
        notes = {k: n for k, n in json.load(open(args.revise)).items() if k in rows}
        revised = ask(args.stage, notes, best)
        revised_failures = failures_of(revised)
        repair(revised, revised_failures, f"{args.stage}_")
        for k, r in revised.items():
            if not revised_failures[k]:  # otherwise the reply from before the revision still stands
                best[k], failures[k] = r, []
        print(f"revised {sum(not f for f in revised_failures.values())} of {len(notes)}", flush=True)

    gold = run_cached(measure, {_fingerprint(k, r): (rows[k], r) for k, r in best.items()}, GOLD, args.workers)
    bad = {k: f for k, f in failures.items() if f}
    print(f"final: {len(bad)} of {len(rows)} problems have failures", flush=True)
    for k, f in sorted(bad.items(), key=lambda x: int(x[0])):
        print(k, f[0][:300], flush=True)
    save_json(args.out, {k: {**r, "gold": gold[_fingerprint(k, r)], "failures": failures[k]} for k, r in best.items()})
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
