"""Compare canonical outputs (see sandbox.encode) and turn raw runs into verdicts.

Runs in the dispatcher, which is the only process that holds expected outputs.
Equality follows Python `==` on the original values, with two relaxations:
floats compare within 1e-5 (the tolerance LeetCode problems state), and lists
and tuples are interchangeable.
"""

import json
import math

from pipeline.tasks.leetcode.verifier.sandbox import CONFINE_FAILED_EXIT, READY

FLOAT_TOL = 1e-5

PASS = "PASS"
WRONG_ANSWER = "WRONG_ANSWER"
RUNTIME_ERROR = "RUNTIME_ERROR"
TIMEOUT = "TIMEOUT"
TOO_SLOW = "TOO_SLOW"
MALFORMED = "MALFORMED"
INFRA_ERROR = "INFRA_ERROR"


def _is_number(x) -> bool:
    return isinstance(x, (int, float))  # bool is an int, matching Python ==


def match(got, exp) -> bool:
    if _is_number(exp) and _is_number(got):
        if isinstance(exp, float) or isinstance(got, float):
            return math.isclose(got, exp, rel_tol=FLOAT_TOL, abs_tol=FLOAT_TOL) or got == exp
        return got == exp
    if isinstance(exp, list):
        return isinstance(got, list) and len(got) == len(exp) and all(map(match, got, exp))
    if isinstance(exp, dict):
        if not isinstance(got, dict) or got.keys() != exp.keys():
            return False
        if "__dict__" in exp:
            g = {json.dumps(k, sort_keys=True): v for k, v in got["__dict__"]}
            e = {json.dumps(k, sort_keys=True): v for k, v in exp["__dict__"]}
            return g.keys() == e.keys() and all(match(g[k], e[k]) for k in e)
        if "__set__" in exp:
            return sorted(map(json.dumps, got["__set__"])) == sorted(map(json.dumps, exp["__set__"]))
        return all(match(got[k], exp[k]) for k in exp)
    return type(got) is type(exp) and got == exp


def judge(run: dict, expected: list) -> dict:
    """Correctness verdict for one sandbox run (timing gate applied by caller).

    expected: decoded canonical outputs, one per test.
    """
    out = run["stdout"]
    base = {"cpu_s": run["cpu_s"], "wall_s": run["wall_s"], "num_tests": len(expected),
            "tests_passed": 0, "first_fail": None, "error": None}
    if not out.startswith(READY):
        if run["outcome"] == "exited" and run["exit_code"] == CONFINE_FAILED_EXIT:
            return {**base, "verdict": INFRA_ERROR, "error": "confine_failed"}
        return {**base, "verdict": INFRA_ERROR, "error": f"no_ready:{run['outcome']}:{run['exit_code']}"}

    lines = out[len(READY):].split(b"\n")
    passed = 0
    for i, exp in enumerate(expected):
        if i >= len(lines) - 1:  # last element is the partial line after the final "\n"
            break
        try:
            rec = json.loads(lines[i])
        except ValueError:
            rec = None
        if not isinstance(rec, dict):
            return {**base, "verdict": RUNTIME_ERROR, "tests_passed": passed, "first_fail": i, "error": "bad_output"}
        if "error" in rec:
            return {**base, "verdict": RUNTIME_ERROR, "tests_passed": passed, "first_fail": i,
                    "error": str(rec["error"])[:64]}
        if not match(rec.get("out"), exp):
            return {**base, "verdict": WRONG_ANSWER, "tests_passed": passed, "first_fail": i}
        passed += 1

    if passed == len(expected) and run["outcome"] == "exited":
        return {**base, "verdict": PASS, "tests_passed": passed}
    if run["outcome"] == "timeout":
        return {**base, "verdict": TIMEOUT, "tests_passed": passed, "first_fail": passed}
    if run["outcome"] == "output_limit":
        return {**base, "verdict": RUNTIME_ERROR, "tests_passed": passed, "first_fail": passed, "error": "output_limit"}
    return {**base, "verdict": RUNTIME_ERROR, "tests_passed": passed, "first_fail": passed,
            "error": f"exit:{run['exit_code']}"}
