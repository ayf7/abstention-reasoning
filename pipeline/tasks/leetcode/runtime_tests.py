"""AI-written runtime tests for leetcode: what Codex is asked for and what makes a reply acceptable.

Each problem gets one generator, a small Python program that builds the actual
test inputs. `series` grow one constraint quantity up to its limit, each aimed at
a named slower approach. `limit_cases` put every quantity at its limit at once.
verifier/runtime.py runs them and compares a program against gold.
"""
import re

MIN_REACH = 0.9  # a series' last size must reach this share of the numeric bound its constraint states
MAX_SERIES, MAX_LIMIT_CASES = 10, 3

PROMPT = """Write runtime tests for this LeetCode problem. They will catch correct submissions whose cost grows faster than the reference solution's, or that are too slow at the largest allowed inputs. Inputs are compared by running the submission and the reference solution on the same data.

Work in this order:

1. List in `approaches` the plausible correct solutions that are asymptotically slower than the reference: for example ones that repeat work, re-scan, re-sort, slice, copy, insert into an ordered list, or recurse without a memo.

2. Classify every numbered constraint below in `constraints`. Use `sweep` for a line that bounds a quantity, and list each quantity it bounds in `quantities`; a line can bound several. Use `rule` for a line that only states a property with no range to grow, such as uniqueness or a fixed length.

3. Write `series`. Every quantity of every `sweep` line needs at least one series, even if neither the reference nor its complexity below depends on that quantity. A series grows one quantity, keeping the others large.
- Choose the input's content by hand for that series. Order, duplicates and value distribution decide whether a slower approach is slow: the same sizes can be fast for it in one order and slow in another. Do not use uniformly random input unless random really is the hard case.
- Name in `targets` the approach from step 1 that the series exposes. A quantity that many approaches depend on should get several series, up to {max_series} series in total, with different content. A quantity that none depends on gets one series, with `targets` set to "none".
- `sizes`: at least 4 increasing integers, about evenly spaced on a log scale, the last being the largest the constraint allows (for a value, the largest magnitude). If fewer than 4 values are allowed, list them all. If no bound is stated, go up to 10^5. A later size must never make the input easier.
- The reference must do its full work at every size. A size the constraint allows can still let it return at once, such as a length it rejects for not being divisible by 3, or content it handles by a shortcut. In that case use the nearest input where it does its full work, such as 19,998 rather than 20,000.
- Scale linked quantities along with the series, for example k = n // 2, so the input stays hard at every size. Do not hold one at a value that hides the growth.
- At the last size, the reference solution should take at most about 2 seconds. Keep the other quantities only as large as that allows, except the swept one, which must reach its limit.

4. Write `limit_cases`: 1 to {max_limit}, each with every quantity at its largest allowed value at the same time, as far as the constraints permit it, on the content hardest for the approaches in step 1, without letting the reference take a shortcut as in step 3. If every quantity at its largest makes the work small because two quantities cancel, such as a pattern as long as the text leaving one window to scan, also write a case with the main size at its largest and the other quantity at the value that makes the reference work hardest. No time cap applies here.

5. Write `generator`, Python source (standard library only; no files, no network) defining:
- `make(rng, name, size)`: returns a dict with every parameter of the starter code, for the series or limit case called `name` at `size` (`size` is None for a limit case). `rng` is a `random.Random`, and all randomness must come from it. Return plain JSON values only. A ListNode or TreeNode argument is LeetCode's list form (a tree in level order, None for a missing child); we convert it.
- `legal(args)`: True only if the dict satisfies every constraint and every guarantee in the statement, including ones like "the input is generated such that ...". Check each one.
- `build(args, solution, harness)`, only when an argument cannot be plain JSON: linked lists inside a list, two lists sharing nodes, an argument that must be a node inside a tree, a custom class, or an API the solution calls as a global. It runs before the clock starts, gets the dict from make (which may carry extra JSON keys for it), and returns the real call arguments. `solution` is the solution's globals, so set what it expects there, such as `solution["isBadVersion"] = ...`. Take node classes from `solution.get("ListNode") or harness["ListNode"]`. With `build`, nothing is converted for you.

Deep recursion is no reason to shrink an input. The reference runs with a recursion limit of 10,000,000, not Python's usual 1000, and with `functools.cache` and `lru_cache(None)` replaced by a Python version. We measured `@cache` recursion working at depth 1,000,000. Recursion through a C function, such as `max(dfs(x) for x in g[i])`, works too; we measured it at depth 500,000.

You may be unable to run code here. We run make, legal and the reference on every input, then send back anything that failed, including sizes where the reference stops doing work. Reason through each input carefully, tracing what the reference and the targeted approach do on it.

If the constraints make a requirement impossible, give your closest attempt and explain in `limitations`. Otherwise leave `limitations` empty; being unable to run code does not belong there.

## Problem
{problem}

## Numbered constraints
{lines}

## Starter code
```python
{starter}
```

## Reference solution
```python
{gold}
```

## Reference solution's complexity
Time: {time}
"""

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["approaches", "constraints", "series", "limit_cases", "generator", "limitations"],
    "properties": {
        "approaches": {"type": "array", "items": {"type": "string"}},
        "constraints": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["line", "kind", "quantities"],
            "properties": {"line": {"type": "integer"}, "kind": {"type": "string", "enum": ["sweep", "rule"]},
                           "quantities": {"type": "array", "items": {"type": "string"}}}}},
        "series": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "line", "quantity", "targets", "sizes"],
            "properties": {"name": {"type": "string"}, "line": {"type": "integer"}, "quantity": {"type": "string"},
                           "targets": {"type": "string"}, "sizes": {"type": "array", "items": {"type": "integer"}}}}},
        "limit_cases": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "targets"],
            "properties": {"name": {"type": "string"}, "targets": {"type": "string"}}}},
        "generator": {"type": "string", "description": "Python source defining make and legal"},
        "limitations": {"type": "string"},
    },
}

_NUMBER = r"\d+(?:\s*\*\s*10\s*\^\s*\d+|\s*\^\s*\d+)?"
_LEADING_BOUND = re.compile(rf"-?\s*{_NUMBER}\s*(?:<=?|≤)")
_TRAILING_BOUND = re.compile(rf"(<=?|≤)\s*({_NUMBER})\s*(?:-\s*1)?\s*\.?$")


def constraint_lines(problem: str) -> list[str]:
    """The lines of the statement's Constraints section."""
    i = problem.rfind("Constraints")
    lines = [x.strip() for x in problem[i:].splitlines()[1:]] if i >= 0 else []
    return [x for x in lines if x and not x.lower().startswith(("follow", "note", "example"))]


def _bound(line: str) -> int | None:
    """The largest integer allowed by the numeric bound that ends a constraint line, if it has one."""
    m = _TRAILING_BOUND.search(line)
    if not m:
        return None
    return eval(re.sub(r"\s", "", m[2].replace("^", "**")), {"__builtins__": {}}) - (m[1] == "<")  # digits, *, ** only


def stated_limit(p: dict, series: dict) -> int | None:
    """The numeric bound on the constraint line this series grows, if it states one."""
    lines = constraint_lines(p["problem"])
    return _bound(lines[series["line"] - 1]) if lines else None


def format_prompt(p: dict) -> str:
    lines = constraint_lines(p["problem"])
    return PROMPT.format(max_series=MAX_SERIES, max_limit=MAX_LIMIT_CASES, problem=p["problem"], starter=p["starter_code"].strip(),
                         gold=p["gold"].strip(), time=p["complexity"]["time"],
                         lines="\n".join(f"{i}. {x}" for i, x in enumerate(lines, 1)) or "(none stated; treat each argument's size as a quantity)")


def validate(p: dict, r: dict) -> None:
    """ValueError unless the reply covers every constraint, reaches each stated limit or explains why not, and has a limit case."""
    lines = constraint_lines(p["problem"])
    if not re.search(r"\bdef make\(", r["generator"]) or not re.search(r"\bdef legal\(", r["generator"]):
        raise ValueError("no make/legal")
    if sorted(c["line"] for c in r["constraints"]) != list(range(1, len(lines) + 1)):
        raise ValueError(f"constraints classified {sorted(c['line'] for c in r['constraints'])} != 1..{len(lines)}")
    names = [s["name"] for s in r["series"]] + [c["name"] for c in r["limit_cases"]]
    if len(names) != len(set(names)):
        raise ValueError(f"repeated names {names}")
    fewest = 1 if any(c["kind"] == "sweep" for c in r["constraints"]) else 0  # a fixed-size input has nothing to grow
    if not fewest <= len(r["series"]) <= MAX_SERIES or not 1 <= len(r["limit_cases"]) <= MAX_LIMIT_CASES:
        raise ValueError(f"{len(r['series'])} series, {len(r['limit_cases'])} limit cases")
    covered = {s["quantity"] for s in r["series"]}  # a quantity can appear on several lines; one series covers it
    for c in r["constraints"]:
        line = lines[c["line"] - 1]
        if c["kind"] == "rule" and (_bound(line) is not None or _LEADING_BOUND.match(line)):
            raise ValueError(f"line {c['line']} has a numeric range but is a rule: {line}")
        if c["kind"] == "sweep" and not c["quantities"]:
            raise ValueError(f"line {c['line']} is a sweep with no quantities: {line}")
        quantities = c["quantities"] if c["kind"] == "sweep" else []
        for q in quantities:
            same = re.search(rf"(?:==\s*{re.escape(q)}\b|(?<![\w.]){re.escape(q)}\s*==)", line)  # n == nums.length is one quantity with two names
            if q not in covered and not (same and (covered & set(quantities) or any(s["line"] == c["line"] for s in r["series"]))):
                raise ValueError(f"no series for {q} on line {c['line']}: {line}")
    for s in r["series"]:
        if s["sizes"] != sorted(set(s["sizes"])) or not s["sizes"]:
            raise ValueError(f"{s['name']} sizes {s['sizes']}")
        if not 1 <= s["line"] <= max(len(lines), 1):
            raise ValueError(f"{s['name']} line {s['line']}")
        bound = stated_limit(p, s)
        if bound is not None and s["sizes"][-1] < MIN_REACH * bound and not r["limitations"].strip():  # other constraints can make a stated limit unreachable
            raise ValueError(f"{s['name']} stops at {s['sizes'][-1]}, constraint allows {bound}: {lines[s['line'] - 1]}")
