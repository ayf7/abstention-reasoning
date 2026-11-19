"""Build data/leetcode_o3 from data/leetcode/problems in one pass.

    python scripts/build_leetcode_o3.py --dst /share/abdelfattah/ayf7/data/leetcode_o3
    python scripts/build_leetcode_o3.py --dst DIR --allow-codex   # ask Codex (paid) for replies missing from the cache

Each row, in every split, is:
  1. made standalone (mode "module"): gold and hint code carry their own imports and
     classes; `setup` keeps only what the tests call to build their arguments;
  2. given limit tests again where gold changed under the new setup;
  3. labelled with its gold's time and memory `complexity` (Codex);
  4. given lost exponents back in 16 statements;
  5. given real correctness tests (`test_entry`) where a constant answer used to pass;
  6. given the pseudocode hint ladder: insights 1-2, rungs 1-3 (Codex), then the gold.

Codex replies are cached under data/.codex_replies by problem index; without
--allow-codex a missing reply is an error, so a rebuild costs nothing. Formatted
prompts are not built; run create_prompts --data-name leetcode_o3 afterwards.
"""
import argparse
import ast
import builtins
import importlib
import json
import random
import re
import symtable
from pathlib import Path

from pipeline.core import codex
from pipeline.core.io import save_json
from pipeline.core.method import DATA_ROOT
from pipeline.tasks.leetcode import limit_tests
from pipeline.tasks.leetcode.task import spec_of
from pipeline.tasks.leetcode.verifier import service

SRC = DATA_ROOT / "leetcode" / "problems"
SPLITS = ("primitives", "sft_train", "sft_val", "rl_train", "rl_val", "eval")
COMPLEXITY_CACHE = DATA_ROOT / ".codex_replies" / f"complexity_v3__{codex.MODEL}"
LADDER_CACHE = DATA_ROOT / ".codex_replies" / f"ladder__{codex.MODEL}"
LADDER_FEWSHOT = Path(__file__).with_name("build_leetcode_o3_fewshot.json")


# ---------- 1. Standalone rows ----------

MISSING_IN_SETUP = {"re": "import re", "randint": "from random import randint"}  # 363 and 686 use these unimported
# Gold that broke or changed under setup's `pow` and `bisect`, or lacked an import: its generated tests must be redone
BROKEN_UNDER_SETUP = re.compile(r"\bpow\(|\bbisect\.|\bre\.|\brandint\(")
NODE_COMMENT = re.compile(r"^# Definition for .*\n?(?:#.*\n?)*", re.M)


def free_names(code: str) -> set[str]:
    """Names the code reads but never defines, builtins excluded."""
    top = symtable.symtable(code, "<gold>", "exec")
    defined = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    names, stack = set(), [top]
    while stack:
        table = stack.pop()
        stack += table.get_children()
        names |= {s.get_name() for s in table.get_symbols() if s.is_global() or (table is top and s.is_referenced())}
    return names - defined - set(dir(builtins))


def bindings(setup: str) -> dict[str, str]:
    """{name: the setup statement that binds it}; a later statement wins, as it does when setup runs."""
    out = {}
    for node in ast.parse(setup).body:
        if isinstance(node, ast.Import):
            out.update({a.asname or a.name: f"import {a.name}" for a in node.names})
        elif isinstance(node, ast.ImportFrom) and node.names[0].name == "*":
            module = importlib.import_module(node.module)
            public = getattr(module, "__all__", [n for n in dir(module) if not n.startswith("_")])
            out.update({n: f"from {node.module} import {n}" for n in public})
        elif isinstance(node, ast.ImportFrom):
            out.update({a.asname or a.name: f"from {node.module} import {a.name}" for a in node.names})
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            out[node.name] = ast.get_source_segment(setup, node)
        elif isinstance(node, ast.Assign):
            out[node.targets[0].id] = ast.get_source_segment(setup, node)
    return out


def preamble(names: set[str], bound: dict[str, str]) -> str:
    """The statements that define `names` and what those statements need themselves, imports first."""
    todo, needed = list(names), set()
    while todo:
        name = todo.pop()
        if name in bound and name not in needed:
            needed.add(name)
            todo += free_names(bound[name])
    imports = sorted({bound[n] for n in needed if bound[n].startswith(("import ", "from "))})
    others = [bound[n] for n in sorted(needed) if not bound[n].startswith(("import ", "from "))]
    return "\n\n".join(filter(None, ["\n".join(imports), *others]))


def standalone(p: dict) -> dict:
    """A row whose gold and hint code run with no setup, and whose setup only builds test arguments."""
    bound = {**MISSING_IN_SETUP, **bindings(p["setup"])}
    if re.search(r"\bbisect\.", p["gold"]):  # setup binds the function, which is what `bisect(...)` wants
        bound["bisect"] = "import bisect"
    names = free_names(p["gold"]) & bound.keys()

    def complete(code: str) -> str:  # hint skeletons need not parse, so look for the names as words
        used = {n for n in names if re.search(rf"\b{n}\b", code)}
        return "\n\n".join(filter(None, [preamble(used, bound), NODE_COMMENT.sub("", code).strip("\n")])) + "\n"

    harness = "from collections import deque\n\n" + p["setup"][p["setup"].index("inf = float('inf')"):]
    return {**p, "mode": "module", "setup": harness, "gold": complete(p["gold"]),
            "hint_steps": [{**h, "code": h["code"] and complete(h["code"])} for h in p["hint_steps"]]}


# ---------- 3. Complexity labels ----------

COMPLEXITY_PROMPT = """State the worst-case time and memory complexity of this reference solution to a LeetCode problem. Everything you need is below; do not use any tools.

Analyze the code exactly as written, not the best possible algorithm for the problem.
- Time: the worst case over all inputs the constraints allow. Count the true cost of each Python operation: a slice, `sum`, `in` on a list, `list.insert`, `list.pop(0)` and string concatenation are linear in their size; `sorted` is n log n; dict and set operations are O(1) on average.
- Integers: arithmetic, comparisons and bit operations cost O(1) and an integer takes O(1) memory while its value fits in 64 bits. Count digits only for values that can grow past 64 bits.
- Memory: the worst-case peak extra memory, including the recursion stack, memo tables and temporary copies. Exclude the input arguments and the returned value, even when the returned value is built up during the run.
- Tightness: give the smallest bound that holds in the worst case. A bound that no allowed input can reach is wrong.
- Write `why_time` and `why_memory` first, then make `time` and `memory` state exactly what they conclude.

Variables: each symbol stands for one quantity from the problem's Constraints section, at its largest. In `variables`, copy that quantity exactly as the constraints write it, such as `nums.length`, `nums[i]`, `k` or `The number of nodes in the tree`.
- Never define other quantities, such as the answer's length, a digit count or a number of distinct values. Bound them using constraint quantities instead: the digits of x become log(x).
- Keep each quantity that affects the growth. Do not replace it with the number from its bound. A quantity the same for every input (such as an alphabet of 26 letters) is a constant and gets no symbol.
- A symbol is a short name made only of letters. Write O(n^2) and list n with the quantity `nums.length`; never write O(nums.length^2).
- If a size has no bound in the constraints, such as the length of a string, use the argument's own name as the quantity, such as `queryIP`.
- List exactly the symbols used in `time` and `memory`.

Format of `time` and `memory`: one plain ASCII expression inside O(), using only symbols, whole numbers, + * ^ and parentheses, and the functions log(), sqrt() and factorial().
- An exponent may be a fraction: O(n^(1/3)), O(3^(n/7)).
- Examples: O(1), O(n), O(n^2*k), O(m*log(n)+k), O(2^n*n), O(sqrt(v)).
- Write every multiplication with *. Do not use conditions, cases, min, max, other notation, words or guards such as +1. If the bound depends on the case, give only the worst case.

## Problem
{problem}

## Reference solution
```python
{gold}
```
"""

COMPLEXITY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["why_time", "why_memory", "time", "memory", "variables"],
    "properties": {
        "why_time": {"type": "string", "description": "at most two sentences"},
        "why_memory": {"type": "string", "description": "at most two sentences"},
        "time": {"type": "string", "description": "e.g. O(n^2*k)"},
        "memory": {"type": "string", "description": "e.g. O(n)"},
        "variables": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["symbol", "quantity"],
            "properties": {"symbol": {"type": "string"}, "quantity": {"type": "string"}}}},
    },
}

FUNCTIONS = {"log", "sqrt", "factorial"}


def symbols(label: str) -> set[str]:
    """The symbols in a label such as O(n^2*log(k)); ValueError if it breaks the format."""
    m = re.fullmatch(r"O\((.+)\)", label)
    if not m or not label.isascii():
        raise ValueError(f"not O(...): {label}")
    try:
        tree = ast.parse(m[1].replace("^", "**"), mode="eval")
    except SyntaxError:
        raise ValueError(f"bad expression: {label}")
    names = set()
    exponents = {id(x) for node in ast.walk(tree) if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) for x in ast.walk(node.right)}
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and id(node) not in exponents:
            raise ValueError(f"division outside an exponent: {label}")
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and len(node.args) == 1 and not node.keywords):
                raise ValueError(f"bad function: {label}")
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Constant):
            if type(node.value) is not int:
                raise ValueError(f"bad number: {label}")
        elif not isinstance(node, (ast.Expression, ast.BinOp, ast.Add, ast.Mult, ast.Pow, ast.Div, ast.Load)):
            raise ValueError(f"bad operator: {label}")
    return names - FUNCTIONS


def check_complexity(problem: str, reply: dict) -> None:
    """ValueError unless the labels are well formed and every symbol is a constraint quantity."""
    used = symbols(reply["time"]) | symbols(reply["memory"])
    listed = {v["symbol"] for v in reply["variables"]}
    if used != listed:
        raise ValueError(f"symbols used {sorted(used)} != listed {sorted(listed)}")
    text = " ".join(problem[problem.rindex("Constraints"):].split())
    for v in reply["variables"]:
        q = " ".join(v["quantity"].split())
        if not re.search(rf"(?<![\w.]){re.escape(q)}(?!\w)", text):
            raise ValueError(f"not in the constraints: {q}")


# ---------- 4. Lost exponents ----------

EXPONENTS = {
    333: [("-228 <= nums1[i], nums2[i], nums3[i], nums4[i] <= 228", "-2^28 <= nums1[i], nums2[i], nums3[i], nums4[i] <= 2^28")],
    399: [("value at most than 230.", "value at most than 2^30.")], 467: [("in the range [1, 210].", "in the range [1, 2^10].")],
    499: [("1 <= nums[i] <\xa0216", "1 <= nums[i] <\xa02^16")], 527: [("0 <= image[i][j], color < 216", "0 <= image[i][j], color < 2^16")],
    599: [("0 <= nums[i] < 216", "0 <= nums[i] < 2^16")], 745: [("0 <= nums[i] < 216", "0 <= nums[i] < 2^16")],
    1150: [("in the range [1, 210].", "in the range [1, 2^10].")], 1280: [("0 <= deliciousness[i] <= 220", "0 <= deliciousness[i] <= 2^20")],
    1325: [("1 <= responses.length <= 102", "1 <= responses.length <= 10^2")], 1337: [("0 <= nums[i] < 210", "0 <= nums[i] < 2^10")],
    1841: [("in the range [1, 214].", "in the range [1, 2^14].")], 1872: [("doesn't exceed\xa0253-1", "doesn't exceed\xa02^53-1")],
    2156: [("1 <= nums[i] <= 230", "1 <= nums[i] <= 2^30")], 2234: [("1 <= nums[i] <= 220 - 1", "1 <= nums[i] <= 2^20 - 1")],
    2303: [("0 <= nums[i] < 230", "0 <= nums[i] < 2^30")]}


def fix_exponents(p: dict) -> dict:
    problem = p["problem"]
    for old, new in EXPONENTS.get(p["index"], []):
        assert problem.count(old) == 1, (p["index"], old, problem.count(old))
        problem = problem.replace(old, new)
    return {**p, "problem": problem}


# ---------- 5. Real correctness tests ----------
# These problems modify an argument (tests saw only the None returned), need a node or an
# API the test text could not express, or had almost no tests. Each gets `test_entry`, a
# wrapper around `entry` that builds the real arguments and returns what must be compared;
# `entry` stays plain because runtime tests call it. Expected outputs come from gold.

IN_PLACE = """
def _in_place(f, name):
    def run(**args):
        f(**args)
        return args[name]
    return run
"""
KEPT = """
def _kept(f, name):
    def run(**args):
        k = f(**args)
        return [k, args[name][:k]]
    return run
"""
WIGGLE = """
def _wiggle(f, strict):
    def run(nums):
        before = sorted(nums)
        f(nums=nums)
        rising = lambda a, b: a < b or (not strict and a == b)
        return sorted(nums) == before and all(rising(a, b) if i % 2 == 0 else rising(b, a) for i, (a, b) in enumerate(zip(nums, nums[1:])))
    return run
"""
NODES = """
def _nodes(root):
    found, stack = {}, [root]
    while stack:
        node = stack.pop()
        if node:
            if node.val in found:
                raise ValueError("statement promises unique values")
            found[node.val] = node
            stack += [node.left, node.right]
    return found
"""
BY_VALUE = NODES + """
def _by_value(f, *names, absent=False):
    def run(root, **args):
        found = _nodes(root)
        if len({args[n] for n in names}) < len(names):
            raise ValueError("statement promises different nodes")
        for n in names:
            args[n] = found.get(args[n]) or TreeNode(args[n]) if absent else found[args[n]]
        return f(root=root, **args)
    return run
"""
TARGET_COPY = NODES + """
def _target_copy(f):
    def run(original, cloned, target):
        node, twin, stack = _nodes(original)[target], None, [(original, cloned)]
        while stack:
            a, b = stack.pop()
            if a:
                twin = b if a is node else twin
                stack += [(a.left, b.left), (a.right, b.right)]
        return f(original=original, cloned=cloned, target=node) is twin
    return run
"""
SHARED = """
def _shared(f):
    def chain(head):
        out = []
        while head:
            out.append(head)
            head = head.next
        return out

    def run(headA, headB):
        a, b = chain(headA), chain(headB)
        n = 0
        while n < min(len(a), len(b)) and a[-1 - n].val == b[-1 - n].val:
            n += 1
        if n == len(b):
            headB = a[len(a) - n]
        elif n:
            b[-1 - n].next = a[len(a) - n]
        before = chain(headA), chain(headB)
        out = f(headA=headA, headB=headB)
        where = None if out is None else next((i for i, x in enumerate(a) if x is out), -1)
        return [where, (chain(headA), chain(headB)) == before]
    return run
"""
CYCLE = """
def _cycle(f):
    def run(head, pos=-1):
        if pos >= 0:
            nodes, node = [], head
            while node:
                nodes.append(node)
                node = node.next
            nodes[-1].next = nodes[pos]
        return f(head=head)
    return run
"""
LISTS = """
def _lists(f):
    def chain(cls, values):
        head = None
        for v in reversed(values):
            head = cls(v, head)
        return head
    return lambda lists: f(lists=[chain(f.__globals__.get("ListNode", ListNode), x) for x in lists])
"""
BAD_VERSION = """
def _bad_version(f):
    def run(n, bad):
        f.__globals__["isBadVersion"] = lambda v: v >= bad
        return f(n=n)
    return run
"""
CATEGORIES = """
def _categories(f):
    def run(n, categoryHandler):
        class Handler:
            def haveSameCategory(self, a, b):
                return 0 <= a < n and 0 <= b < n and categoryHandler[a] == categoryHandler[b]
        return f(n=n, categoryHandler=Handler())
    return run
"""
RANDOM_TREE = """
def _random_tree(f):
    class Node:
        def __init__(self, val=0, left=None, right=None, random=None):
            self.val, self.left, self.right, self.random = val, left, right, random

    class NodeCopy(Node):
        pass

    def walk(root):
        order, i = [root] if root else [], 0
        while i < len(order):
            if order[i]:
                order += [order[i].left, order[i].right]
            i += 1
        while order and order[-1] is None:
            order.pop()
        return order

    def run(root):
        f.__globals__.setdefault("Node", Node)
        f.__globals__.setdefault("NodeCopy", NodeCopy)
        made = {i: Node(x[0]) for i, x in enumerate(root) if x}
        i = 1
        for parent in [made[j] for j in sorted(made)]:
            for side in ("left", "right"):
                if i < len(root):
                    setattr(parent, side, made.get(i))
                    i += 1
        for j, node in made.items():
            node.random = made.get(root[j][1])
        order = walk(f(root=made.get(0)))
        at = {id(x): j for j, x in enumerate(order) if x}
        return [[[x.val, None if x.random is None else at.get(id(x.random), -1)] if x else None for x in order],
                not any(id(x) in at for x in made.values())]
    return run
"""

RETURN_ONLY = {25: "nums", 38: "matrix", 61: "matrix", 63: "nums", 74: "nums1", 82: "root", 97: "root", 110: "board", 119: "head",
               146: "s", 148: "nums", 212: "nums", 213: "rooms", 215: "board", 255: "s", 834: "arr"}
WRAPPERS = {**{k: (IN_PLACE, f"_in_place({{}}, {v!r})") for k, v in RETURN_ONLY.items()},
            22: (KEPT, "_kept({}, 'nums')"), 66: (KEPT, "_kept({}, 'nums')"),
            210: (WIGGLE, "_wiggle({}, False)"), 237: (WIGGLE, "_wiggle({}, True)"),
            182: (BY_VALUE, "_by_value({}, 'p', 'q')"), 1232: (BY_VALUE, "_by_value({}, 'p', 'q', absent=True)"),
            1204: (BY_VALUE, "_by_value({}, 'u')"), 1041: (TARGET_COPY, "_target_copy({})"), 132: (SHARED, "_shared({})"),
            118: (CYCLE, "_cycle({})"), 19: (LISTS, "_lists({})"), 209: (BAD_VERSION, "_bad_version({})"),
            2117: (CATEGORIES, "_categories({})"), 1117: (RANDOM_TREE, "_random_tree({})"), 363: ("", "{}")}


def cycles(rng):
    sizes = [2, 3, 5, 10, 50, 100, 1000, 10000, 10000, 10000] + [rng.randint(2, 500) for _ in range(10)]
    pos = lambda n: rng.choice([-1, 0, n - 1, rng.randrange(n)])
    return [f"head=list_node({[rng.randint(-10**5, 10**5) for _ in range(n)]}), pos={pos(n)}" for n in sizes]


def merges(rng):
    def lists(k, total):
        cuts = [sorted(rng.randint(-10**4, 10**4) for _ in range(total // k)) for _ in range(k)]
        return f"lists={cuts}"
    return [lists(10**4, 10**4), lists(2, 10**4), lists(1, 10**4), lists(100, 10**4)] + [lists(rng.randint(1, 8), rng.randint(8, 60)) for _ in range(16)]


def versions(rng):
    top = 2**31 - 1
    pairs = [(1, 1), (2, 1), (2, 2), (top, 1), (top, top), (top, top // 2), (top, top - 1)]
    for _ in range(23):
        n = rng.randint(1, 10**rng.randint(1, 9))
        pairs.append((n, rng.randint(1, n)))
    return [f"n={n}, bad={b}" for n, b in pairs]


def categories(rng):
    tests = [list(range(100)), [0] * 100, [i % 2 for i in range(100)], [0]]
    tests += [[rng.randrange(rng.randint(1, n)) for _ in range(n)] for n in (rng.randint(1, 100) for _ in range(16))]
    return [f"n={len(c)}, categoryHandler={c}" for c in tests]


def random_trees(rng):
    def tree(n):
        rows, parents, slots = [[rng.randint(1, 10**6), None]], 1, 2
        while parents < n and slots:
            slots -= 1
            if rng.random() < 0.75:
                rows.append([rng.randint(1, 10**6), None])
                parents, slots = parents + 1, slots + 2
            else:
                rows.append(None)
        filled = [i for i, x in enumerate(rows) if x]
        for x in filter(None, rows):
            x[1] = rng.choice(filled + [None])
        return f"root={rows}"
    return [tree(n) for n in [1, 2, 3, 1000, 1000, 500, 100, 50] + [rng.randint(2, 30) for _ in range(12)]]


def boards(rng):
    def board(n):
        out = ""
        while len(out) < n:
            c = rng.choice("RYBGW")
            out += c if out[-2:] != c * 2 else ""
        return out
    pairs = [("WRRBBW", "RB"), ("WWRRBBWW", "WRBRW"), ("G", "GGGGG"), ("RBYYBBRRB", "YRBGB")]
    pairs += [(board(rng.randint(1, 16)), "".join(rng.choice("RYBGW") for _ in range(rng.randint(1, 5)))) for _ in range(16)]
    return [f"board={b!r}, hand={h!r}" for b, h in pairs]


NEW_TESTS = {118: cycles, 19: merges, 209: versions, 2117: categories, 1117: random_trees, 363: boards}
REPLACED = {209, 2117, 1117, 363}  # their old tests break the statement's rules or cannot be run


def real_tests(p: dict, pool) -> dict:
    """{field: new value} for one problem in WRAPPERS."""
    rng = random.Random(p["index"])
    helper, wrap = WRAPPERS[p["index"]]
    new = NEW_TESTS.get(p["index"], lambda _: [])(rng)
    old = [] if p["index"] in REPLACED else [t for t in p["tests"] if t not in new]
    tests = old + new
    entry, setup = wrap.format(p["entry"]), p["setup"] + helper
    runs = pool.gold_outputs({**spec_of(p), "setup": setup, "entry": entry, "tests": tests})
    keep = [(t, g["expected"]) for t, g in zip(tests, runs) if g["expected"] is not None and not (p["index"] == 237 and g["expected"] == "false")]
    limit = p.get("limit_tests") or 0
    generated = set(p["tests"][len(p["tests"]) - limit:]) if limit else set()
    print(f"  {p['index']}: tests {len(p['tests'])} -> {len(keep)}", flush=True)
    return {"test_entry": entry, "setup": setup, "tests": [t for t, _ in keep], "expected": [e for _, e in keep],
            "limit_tests": sum(t in generated for t, _ in keep)}


# ---------- 6. Pseudocode hint ladder ----------
# One step list per problem, rendered three ways so the rungs cannot disagree:
# rung 1 = every step's "hi"; rung 2 = "mid" for core steps, "hi" otherwise; rung 3 = every "lo".

LADDER_SCHEMA = {"type": "object", "properties": {"gradient": {"type": "string"}}, "required": ["gradient"], "additionalProperties": False}
LADDER_PROMPT = """You are writing a pseudocode hint ladder for a programming problem. A student model sees the problem, two insights, then ONE rung of this ladder, and must then write the code. The rungs, weakest to strongest:

  rung 1: high-level pseudocode of the whole algorithm; the key step (the trick) is only its intent, in one or two lines
  rung 2: the same block, but the key step is spelled out line by line, in words, using the named variables
  rung 3: detailed pseudocode, every line exact
  (after rung 3 comes the gold solution itself)

The rungs must agree with each other and with the gold solution, so you write them as ONE step list that is rendered three ways.

## Format
A JSON object {{"name", "input", "output", "steps"}}:
- name: CamelCase name of the algorithm. input: list of lines "name: meaning", one per input. output: one line, what is returned.
- steps: in order. A step is {{"d": depth, "core": bool, "hi": ..., "mid": ..., "lo": ...}} or {{"d": depth, "note": text}}.
  "hi", "mid" and "lo" are a string or a list of lines; inner lines are indented by 4 spaces per level, relative to the step's depth d.
  - lo: exact pseudocode that follows the gold solution line by line (assignments with <-, FOR/WHILE/IF ... THEN/ELSE/RETURN, mod, div, AND/OR/NOT, =, ≠).
  - hi: the same step as high-level pseudocode statements. For a non-core step, keep its control structure (FOR EACH / WHILE / IF / RETURN) and name the variables that core steps use. For a core step, give only its intent in one or two lines: what it achieves, with no loop structure or line-level detail.
  - mid (core steps only): the lines of lo, ONE FOR ONE (same number of lines, same indentation), with the same variable names, each said in words: "Add one to left", not "left <- left + 1". Indexing such as nums[left] is fine.
  - core: true for the lines that carry the trick, usually ONE step covering the whole key loop or block. Setup, input decoding, simple bookkeeping and reading out the answer are not core.
  - note: a short state definition or reason, shown in every rung. Use sparingly.
  Rendering: rung 1 = every hi; rung 2 = mid for core steps, hi otherwise; rung 3 = every lo. Notes appear in all rungs.
- No Python syntax in any line, notes included: no **, //, %, ==, !=, ^, negative indexing, def, elif, or lines ending in ":". Write a power as "2 to the power k" (^ reads as XOR in Python); a quoted character such as '%' is fine.
- The algorithm, its data structures and its complexity must be the gold solution's. The final step returns the answer.

## Examples
{examples}

## Problem: {title}
{problem}

Insights the student sees before your rung:
{insights}

Gold solution:
```python
{gold}
```

Reply with the JSON object, as text, in `gradient`."""


def _as_lines(x) -> list[str]:
    return [x] if isinstance(x, str) else x


def render(ex: dict, rung: int) -> str:
    """One algorithm block: name, inputs, output, then the steps (ending in RETURN)."""
    out = [f"ALGORITHM {ex['name']}", "INPUT"] + [f"    {x}" for x in ex["input"]]
    out += ["OUTPUT", f"    {ex['output']}", "STEPS"]
    for s in ex["steps"]:
        pad = "    " * (s["d"] + 1)
        if "note" in s:
            out.append(f"{pad}Note: {s['note']}")
            continue
        key = "lo" if rung == 3 else "mid" if rung == 2 and s["core"] else "hi"
        out += [pad + line for line in _as_lines(s[key])]
    return "\n".join(out)


def check_ladder(ex: dict) -> None:
    """ValueError unless ex has the fields render needs."""
    for key, kind in (("name", str), ("input", list), ("output", str), ("steps", list)):
        if not isinstance(ex.get(key), kind) or not ex[key]:
            raise ValueError(f"missing or bad {key}")
    for s in ex["steps"]:
        if not isinstance(s.get("d"), int) or s["d"] < 0:
            raise ValueError(f"bad depth: {s}")
        if "note" in s:
            continue
        need = ("hi", "lo", "mid") if s.get("core") else ("hi", "lo")
        if not isinstance(s.get("core"), bool) or any(not s.get(k) for k in need):
            raise ValueError(f"step lacks core/{'/'.join(need)}: {str(s)[:200]}")


def ladder_prompt(p: dict, examples: str) -> str:
    i1, i2 = p["hint_steps"][0]["insight"], p["hint_steps"][1]["insight"]
    return LADDER_PROMPT.format(examples=examples, title=p["task_id"], problem=p["problem"], gold=p["gold"],
                                insights=f"Insight 1: {i1}\nInsight 2: {i2}")


def with_ladder(p: dict, ladder: dict) -> dict:
    """hint_steps (insight, insight, skeleton, gold) -> both insights, rungs 1-3, gold."""
    i1, i2, _skeleton, gold = p["hint_steps"]
    step = lambda insight=None, code=None, pseudocode=None: {"insight": insight, "code": code, "pseudocode": pseudocode}
    return {**p, "hint_steps": [step(insight=[i1["insight"], i2["insight"]]),
                                *(step(pseudocode=render(ladder, r)) for r in (1, 2, 3)),
                                step(code=gold["code"])]}


# ---------- Build ----------

def replies(prompt_of, keys: list[str], schema: dict, cache: Path, check, allow_codex: bool) -> dict[str, dict]:
    """{key: parsed reply}, from the cache; Codex (paid) is asked for missing keys only with allow_codex."""
    missing = [k for k in keys if not (cache / f"{k}.json").exists()]
    if missing and not allow_codex:
        raise SystemExit(f"{len(missing)} replies missing from {cache}; --allow-codex asks Codex (paid): {missing[:20]}")
    if missing:
        codex.ask_all({k: prompt_of(k) for k in missing}, schema, cache, check)
    if still := [k for k in keys if not (cache / f"{k}.json").exists()]:
        raise SystemExit(f"{len(still)} replies still missing from {cache}; rerun to retry: {still[:20]}")
    return {k: json.loads(json.loads((cache / f"{k}.json").read_text())["reply"]) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC, help="base problems (default: data/leetcode/problems)")
    ap.add_argument("--dst", type=Path, required=True, help="output dataset directory; problems go to DST/problems")
    ap.add_argument("--allow-codex", action="store_true", help="ask Codex (paid) for replies missing from the cache")
    args = ap.parse_args()
    if (args.dst / "problems").exists():
        raise SystemExit(f"{args.dst}/problems already exists")

    # Everything that depends on the whole corpus is computed once, keyed by index, from primitives.
    base = {p["index"]: p for p in json.load(open(args.src / "primitives.json"))}
    keys = [str(i) for i in base]
    complexity = replies(lambda k: COMPLEXITY_PROMPT.format(problem=base[int(k)]["problem"], gold=base[int(k)]["gold"]),
                         keys, COMPLEXITY_SCHEMA, COMPLEXITY_CACHE, lambda k, r: check_complexity(base[int(k)]["problem"], r), args.allow_codex)
    print(f"complexity: {len(complexity)} labels", flush=True)

    redo = [limit_tests._strip(standalone(p)) for p in base.values() if BROKEN_UNDER_SETUP.search(p["gold"])]
    added, why, gold_s = limit_tests.generate(redo)
    limit_tests.report(why, gold_s)
    rows = {i: standalone(base[i]) for i in base}
    del base
    rows = {i: limit_tests.append(p, added) if i in added else p for i, p in rows.items()}

    pool = service.VerifierPool(service.physical_cores())
    tests = {i: real_tests(rows[i], pool) for i in sorted(WRAPPERS)}
    pool.close()

    fewshot = {str(ex["index"]): ex for ex in json.load(open(LADDER_FEWSHOT))}
    examples = "\n\n".join(
        f"### {rows[int(k)]['task_id']}\nGold solution:\n```python\n{rows[int(k)]['gold']}\n```\nStep list:\n"
        f"{json.dumps({f: ex[f] for f in ('name', 'input', 'output', 'steps')}, indent=1, ensure_ascii=False)}\n"
        f"Rendered rung 2:\n```\n{render(ex, 2)}\n```" for k, ex in fewshot.items())
    asked = [k for k in keys if k not in fewshot]
    ladders = {k: json.loads(r["gradient"]) for k, r in replies(
        lambda k: ladder_prompt(rows[int(k)], examples), asked, LADDER_SCHEMA, LADDER_CACHE,
        lambda k, r: check_ladder(json.loads(r["gradient"])), args.allow_codex).items()}
    ladders |= fewshot  # problems used as examples keep their hand-written ladder
    print(f"ladder: {len(ladders)} step lists ({len(fewshot)} hand-written)", flush=True)
    del rows

    for split in SPLITS:
        out = []
        for p in json.load(open(args.src / f"{split}.json")):
            i = p["index"]
            p = standalone(p)
            p = limit_tests.append(p, added) if i in added else p
            p = fix_exponents({**p, "complexity": complexity[str(i)]})
            out.append(with_ladder({**p, **tests.get(i, {})}, ladders[str(i)]))
        save_json(args.dst / "problems" / f"{split}.json", out)
        print(f"{args.dst}/problems/{split}.json: {len(out)}", flush=True)
        del out


if __name__ == "__main__":
    main()
