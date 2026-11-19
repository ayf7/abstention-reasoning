"""Build data/leetcode_o2: data/leetcode/problems plus the complexity of each gold solution.

    python scripts/build_leetcode_o2.py                    # annotate everything, then write the dataset
    python scripts/build_leetcode_o2.py --only 2582,2529   # annotate these indices and print; writes no dataset

Rows keep their indices, splits and tests, and gain a `complexity` field. They
are also made standalone (mode "module"): gold and the code in the hints carry
their own imports and class definitions, and `setup` keeps only what the tests
call to build their arguments. Replies are cached, so a rerun only asks for what
is missing; the dataset is written only once every problem has a reply. Formatted
prompts are not copied; rebuild them with create_prompts --data-name leetcode_o2.
"""
import argparse
import ast
import builtins
import importlib
import json
import os
import re
import symtable

from pipeline.core import codex
from pipeline.core.io import save_json
from pipeline.core.method import DATA_ROOT
from pipeline.tasks.leetcode import limit_tests

SRC, DST = DATA_ROOT / "leetcode" / "problems", DATA_ROOT / "leetcode_o2"
STORE = "/share/abdelfattah/ayf7/data/leetcode_o2"  # data/leetcode_o2 links here, as data/leetcode_full does
REPLIES = DATA_ROOT / ".codex_replies" / f"complexity_v3__{codex.MODEL}"
SPLITS = ("primitives", "sft_train", "sft_val", "rl_train", "rl_val", "eval")

PROMPT = """State the worst-case time and memory complexity of this reference solution to a LeetCode problem. Everything you need is below; do not use any tools.

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

SCHEMA = {
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
MISSING_IN_SETUP = {"re": "import re", "randint": "from random import randint"}  # 363 and 686 use these unimported
# Gold that broke or changed under setup's `pow` and `bisect`, or lacked an import: its generated tests must be redone
BROKEN_UNDER_SETUP = re.compile(r"\bpow\(|\bbisect\.|\bre\.|\brandint\(")
NODE_COMMENT = re.compile(r"^# Definition for .*\n?(?:#.*\n?)*", re.M)


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


def constraints_of(problem: str) -> str:
    """The Constraints section with whitespace collapsed."""
    return " ".join(problem[problem.rindex("Constraints"):].split())


def validate(problem: str, reply: dict) -> None:
    """ValueError unless the labels are well formed and every symbol is a constraint quantity."""
    used = symbols(reply["time"]) | symbols(reply["memory"])
    listed = {v["symbol"] for v in reply["variables"]}
    if used != listed:
        raise ValueError(f"symbols used {sorted(used)} != listed {sorted(listed)}")
    text = constraints_of(problem)
    for v in reply["variables"]:
        q = " ".join(v["quantity"].split())
        if not re.search(rf"(?<![\w.]){re.escape(q)}(?!\w)", text):
            raise ValueError(f"not in the constraints: {q}")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated problem indices; prints the replies and writes no dataset")
    ap.add_argument("--sessions", type=int, default=codex.SESSIONS)
    args = ap.parse_args()
    if not args.only and (DST / "problems").exists():
        raise SystemExit(f"{DST}/problems already exists")

    rows = json.load(open(SRC / "primitives.json"))
    prompts = {str(p["index"]): PROMPT.format(problem=p["problem"], gold=p["gold"]) for p in rows}
    statements = {str(p["index"]): p["problem"] for p in rows}
    del rows
    if args.only:
        prompts = {k: prompts[k] for k in args.only.split(",")}
    replies = codex.ask_all(prompts, SCHEMA, REPLIES, lambda k, r: validate(statements[k], r), args.sessions)
    missing = sorted(set(prompts) - set(replies), key=int)
    if args.only:
        for k, r in replies.items():
            print(k, json.dumps(json.loads(r), indent=1))
    if missing:
        raise SystemExit(f"{len(missing)} problems have no reply; rerun to retry: {missing[:20]}")
    if args.only:
        return

    os.makedirs(f"{STORE}/problems", exist_ok=True)
    if not DST.exists():
        DST.symlink_to(STORE)
    redo = [limit_tests._strip(standalone(p)) for p in json.load(open(SRC / "primitives.json")) if BROKEN_UNDER_SETUP.search(p["gold"])]
    added, why, gold_s = limit_tests.generate(redo)
    limit_tests.report(why, gold_s)
    for split in SPLITS:
        rows = [standalone(p) for p in json.load(open(SRC / f"{split}.json"))]
        save_json(DST / "problems" / f"{split}.json", [{**(limit_tests.append(p, added) if p["index"] in added else p),
                                                        "complexity": json.loads(replies[str(p["index"])])} for p in rows])
        print(f"{DST}/problems/{split}.json: {len(rows)}", flush=True)
        del rows


if __name__ == "__main__":
    main()
