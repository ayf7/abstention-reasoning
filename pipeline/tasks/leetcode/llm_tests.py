"""Tests written by an LLM: edge cases, typical inputs and limit-size inputs.

The model writes each test as a dict of argument expressions, so one line can
describe a 10^5-element input (`[10**9] * 10**5`). Expressions run with a few
builtins and a seeded `rng`. Inputs that break the parsed constraints
(limit_tests.Spec.check) are dropped and gold labels the rest: the model never
states an expected output, so it decides what is tested, not what is correct.
When the constraints do not parse, only argument types are checked; where they
do parse, 3 of 9,653 model inputs broke them.
"""
import json
import os
import random
import re
import signal
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from pipeline.tasks.leetcode import limit_tests

MODEL = "gpt-5.6-sol_2026-07-09"
API = "https://tghub.tanyagoyal.com/v1/chat/completions"
N_TESTS = 50
TRIES = 8  # the gateway does not stream and cuts replies at ~100 s, more often under load
CONCURRENCY = 4  # more requests at once make the gateway time out far more often

PROMPT = """You are writing the hidden test suite for a LeetCode problem. A submission is accepted only if it is right on every test within the time limit, so the suite must catch both wrong and slow solutions.

Write {n} tests:
- about 25 "edge": boundary and degenerate inputs where wrong solutions fail: smallest sizes, extreme values, all elements equal, sorted and reverse-sorted input, answers at the first or last position, cases with no valid choice, values large enough to overflow 32 bits. Above all, think about which wrong approaches a contestant would plausibly try on THIS problem (a greedy that is not optimal, a missed case, an off-by-one) and write inputs that break each of them.
- about 15 "common": typical inputs of small to moderate size.
- about 10 "limit": inputs at or near the maximum sizes and values allowed, shaped to be the worst case for slow solutions (not only random).

Every test must satisfy all constraints and every guarantee in the statement (for example "the input is generated such that ..."). Do not write expected outputs.

Each argument value is a Python expression that evaluates to the argument. Expressions may use literals, comprehensions, range, len, list, sorted, reversed, min, max, sum, abs, str, int, chr, ord, zip, enumerate, and rng (a random.Random instance). Use plain literals for small inputs. Keep each expression under 300 characters.{node_note}

## Problem
{problem}

## Starter code
```python
{starter}
```

Reply with JSON only: {{"tests": [{{"kind": "edge" | "common" | "limit", "why": "<a few words>", "args": {{"<parameter>": "<python expression>", ...}}}}, ...]}}
"""

NODE_NOTE = (" A ListNode or TreeNode argument is written as the list LeetCode uses for it (a tree in level order,"
             " None for a missing child); it is converted to nodes for you.")

BUILTINS = {f.__name__: f for f in (range, len, list, sorted, reversed, min, max, sum, abs, str, int,
                                     chr, ord, zip, enumerate, tuple, set, dict, map, bool)}
EVAL_SECONDS = 20  # per problem, for all its expressions


def ask(p: dict) -> str:
    """The model's raw reply for problem `p`."""
    prompt = PROMPT.format(n=N_TESTS, problem=p["problem"], starter=p["starter_code"].strip(),
                           node_note=NODE_NOTE if limit_tests.node_args(p["starter_code"]) else "")
    req = urllib.request.Request(
        API, data=json.dumps({"model": MODEL, "reasoning_effort": "high",
                              "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0",
                 "Authorization": f"Bearer {os.environ['OPENAI_API_KEY_GOYAL']}"})
    for attempt in range(TRIES):  # long high-effort replies sometimes hit the gateway's timeout (5xx); it also rate-limits (429)
        try:
            return json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if (e.code < 500 and e.code != 429) or attempt == TRIES - 1:
                raise
            time.sleep(30 * (attempt + 1))


def parse(reply: str) -> list[dict]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())
    tests = json.loads(text)["tests"]
    assert all(isinstance(t["args"], dict) for t in tests)
    return tests


def _type_error(types: dict[str, str], nodes: dict[str, str], args: dict) -> str | None:
    """Why `args` has a wrong argument type, for the types limit_tests knows and node list forms."""
    for a, t in types.items():
        v = args[a]
        if a in nodes:
            if not (isinstance(v, list) and all(x is None or type(x) is int for x in v)):
                return f"{a} is not a list form of {t}"
        elif t in limit_tests.TYPES and not limit_tests._typed(v, t):
            return f"{a} is not {t}"
    return None


def _alarm(*_):
    raise TimeoutError


def materialize(p: dict, tests: list[dict]) -> list[tuple[dict, str | None, str | None]]:
    """(test, rendered input or None, why it was dropped) per test, in order.

    Runs model-written expressions: call it in a worker process, not the main one.
    """
    nodes = limit_tests.node_args(p["starter_code"])
    try:
        spec = limit_tests.Spec(p["problem"], p["starter_code"])
        names, check = list(spec.args), spec.check
    except Exception:  # constraints the grammar cannot read (as in limit_tests, a half-matched line counts)
        types = dict(limit_tests.params(p["starter_code"]))
        names, check = list(types), lambda args: _type_error(types, nodes, args)
    have, out = set(p["tests"]), []
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(EVAL_SECONDS)
    try:
        for j, t in enumerate(tests):
            rng = random.Random(p["index"] * 1000 + j)
            try:
                args = {a: eval(t["args"][a], {"__builtins__": BUILTINS, "rng": rng}) for a in names}
            except TimeoutError:
                raise
            except Exception as e:
                out.append((t, None, f"expression: {type(e).__name__}"))
                continue
            try:
                why = check(args)
            except TimeoutError:
                raise
            except Exception as e:  # a bound the grammar half-matched; the test cannot be checked
                why = f"check {type(e).__name__}"
            if why:
                out.append((t, None, f"constraint: {why}"))
                continue
            s = limit_tests.render(args, nodes)
            out.append((t, None, "duplicate") if s in have else (t, s, None))
            have.add(s)
    except TimeoutError:
        out += [(t, None, "expression: timeout") for t in tests[len(out):]]
    finally:
        signal.alarm(0)
    return out


def _reply(p: dict, cache: Path) -> str | None:
    f = cache / f"{p['question_id']}.txt"
    if not f.exists():
        try:
            text = ask(p)
        except Exception as e:
            print(f"  ask failed for #{p['question_id']}: {e!r}", flush=True)
            return None
        tmp = f.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.rename(f)
    return f.read_text()


def _materialize(args):
    return materialize(*args)


def generate(problems: list[dict], cache: Path) -> tuple[dict, Counter]:
    """{index: (tests, expected)} of gold-labelled LLM tests, and why tests were dropped.

    Replies are cached in `cache` by question id, so an interrupted run resumes.
    """
    from pipeline.tasks.leetcode.task import spec_of
    from pipeline.tasks.leetcode.verifier import service

    cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(CONCURRENCY) as ex:
        replies = list(ex.map(lambda p: _reply(p, cache), problems))
    why, tests = Counter(), []
    for r in replies:
        try:
            tests.append(parse(r) if r else [])
            why["reply: ok" if r else "reply: none"] += 1
        except Exception:
            tests.append([])
            why["reply: unparsable"] += 1
    cpus = service.physical_cores()
    with ProcessPoolExecutor(len(cpus)) as ex:
        mats = list(ex.map(_materialize, zip(problems, tests)))
    pool = service.VerifierPool(cpus)
    added = {}
    try:
        for p, m in zip(problems, mats):
            why.update(w for _, _, w in m if w)
            new = [s for _, s, _ in m if s]
            runs = pool.gold_outputs({**spec_of(p), "tests": new}) if new else []
            keep = [(t, g["expected"]) for t, g in zip(new, runs) if g["expected"] is not None]
            why["gold failed"] += len(new) - len(keep)
            added[p["index"]] = ([t for t, _ in keep], [e for _, e in keep])
    finally:
        pool.close()
    return added, why
