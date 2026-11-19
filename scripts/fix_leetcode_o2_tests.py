"""Give leetcode_o2 real correctness tests on the problems a constant answer used to pass.

Those problems modify an argument (tests saw only the None returned), need a node
or an API the test text could not express, or had almost no tests. Each now gets
`test_entry`, a wrapper around `entry` that builds the real arguments and returns
what must be compared. `entry` itself stays plain because runtime tests call it.
Expected outputs come from running gold; tests gold rejects are dropped.

    python scripts/fix_leetcode_o2_tests.py
"""
import json
import os
import random
import re

from pipeline.tasks.leetcode.task import spec_of
from pipeline.tasks.leetcode.verifier import service

ROOT = "data/leetcode_o2/problems"
SPLITS = ("primitives", "sft_train", "sft_val", "rl_train", "rl_val", "eval")

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


def fixed_row(p, pool, rng):
    """{field: new value} for one problem."""
    helper, wrap = WRAPPERS[p["index"]]
    new = NEW_TESTS.get(p["index"], lambda _: [])(rng)
    old = [] if p["index"] in REPLACED else [t for t in p["tests"] if t not in new]
    tests = old + new
    entry, setup = wrap.format(p["entry"]), p["setup"] + helper
    runs = pool.gold_outputs({**spec_of(p), "setup": setup, "entry": entry, "tests": tests})
    keep = [(t, g["expected"]) for t, g in zip(tests, runs) if g["expected"] is not None and not (p["index"] == 237 and g["expected"] == "false")]
    limit = p.get("limit_tests") or 0
    generated = set(p["tests"][len(p["tests"]) - limit:]) if limit else set()
    out = {"test_entry": entry, "setup": setup, "tests": [t for t, _ in keep], "expected": [e for _, e in keep],
           "limit_tests": sum(t in generated for t, _ in keep)}
    print(p["index"], p["entry"][11:], f"tests {len(p['tests'])} -> {len(keep)}, distinct outputs {len({e for _, e in keep})}, "
          f"largest expected {max(len(e) for _, e in keep)} chars", flush=True)
    return out


def main():
    rows = {p["index"]: p for p in json.load(open(f"{ROOT}/primitives.json"))}
    if any("test_entry" in rows[k] for k in WRAPPERS):
        raise SystemExit("already fixed")
    pool = service.VerifierPool(service.physical_cores())
    fixes = {k: fixed_row(rows[k], pool, random.Random(k)) for k in sorted(WRAPPERS)}
    pool.close()
    for split in SPLITS:
        path = f"{ROOT}/{split}.json"
        out = [{**p, **fixes.get(p["index"], {})} for p in json.load(open(path))]
        with open(path + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(path + ".tmp", path)
        print(path, sum(p["index"] in fixes for p in out), flush=True)


if __name__ == "__main__":
    main()
