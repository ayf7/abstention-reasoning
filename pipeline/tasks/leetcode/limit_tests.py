"""Parse LeetCode constraint text into an input spec and sample inputs from it.

A spec covers what the constraint lines state formally: argument types, length
and value ranges (possibly bounded by other arguments), alphabets, and a few
structural tags (distinct, sorted, tree, permutation, prime). A line the grammar
does not understand makes the problem unsupported; it is never guessed at.

Statements carry powers in caret form ("10^5", "2^31 - 1"), restored by
statement.repair_statement. Reading a bound too small is safe (inputs stay
valid, only less demanding); too large is caught when gold fails on it.

The original LeetCode tests are tiny, so a brute-force solution passes them.
`main` appends generated tests (gold run on each) to every problem file of a
corpus, in place, and records their count as `limit_tests`:

    python -m pipeline.tasks.leetcode.limit_tests data/leetcode/problems
"""
import argparse
import json
import random
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

SMALL, MAX = 10, 3   # generated inputs per problem
MAX_GOLD_S = 0.8     # drop a max input whose gold run is slower: with ~3 per problem the
                     # 3x-gold limit stays under the verifier's 10 s timeout

LIST_CAP = 200_000   # elements per list argument (rows * cols for matrices)
STR_CAP = 1_000_000  # characters per string argument
PRIME_CAP = 10_000_000

TYPES = {"int", "str", "List[int]", "List[str]", "List[List[int]]"}
LOWER, UPPER = "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTERS = [  # first match wins
    (r"lowercase and uppercase|uppercase and lowercase|(?<!lowercase )(?<!uppercase )English letters", LOWER + UPPER),
    (r"lowercase", LOWER),
    (r"uppercase", UPPER),
]
CMP = {"<=": (0, 1), "<": (1, 1), ">=": (0, -1), ">": (1, -1)}  # (strict, direction)
LOCAL = ("pos:", "elem:", "char:")  # keys that vary per element, not per input


class Unsupported(Exception):
    pass


class _Retry(Exception):
    pass


def fix_numbers(s: str) -> str:
    s = s.replace("\xa0", " ").replace("​", "")
    s = s.replace("≤", "<=").replace("≥", ">=").replace("−", "-").replace("–", "-")
    s = re.sub(r"\b(\w+)\s*\^\s*(-?\w+)", r"(\1**\2)", s)
    return s.strip().rstrip(".").strip()


def signature(starter: str) -> dict[str, str]:
    m = re.search(r"def \w+\(self,?(.*?)\)\s*(->|:)", starter, re.S)
    if not m:
        raise Unsupported("no signature")
    args = {}
    for part in re.split(r",\s*(?=\w+\s*:)", " ".join(m.group(1).split())):
        if not part.strip():
            continue
        name, _, t = part.partition(":")
        t = t.strip().replace(" ", "")
        if t not in TYPES:
            raise Unsupported(f"type {t or '?'}")
        args[name.strip()] = t
    return args


@lru_cache(maxsize=None)
def primes_upto(n: int) -> tuple[int, ...]:
    sieve = bytearray([1]) * (n + 1)
    sieve[:2] = b"\0\0"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return tuple(i for i, v in enumerate(sieve) if v)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):  # deterministic below 3.3e24
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _random_prime(rng, lo, hi):
    for _ in range(10_000):
        x = rng.randint(lo, hi)
        if _is_prime(x):
            return x
    raise _Retry


def _pick(cands):
    """Next quantity to sample: fully bounded first, then partially, then upper-bounded only."""
    for tier in (lambda c: c[1] is not None and c[2] is not None and c[3],
                 lambda c: c[1] is not None and c[2] is not None,
                 lambda c: c[2] is not None):
        ready = sorted(c for c in cands if tier(c))
        if ready:
            k, lo, hi, _ = ready[0]
            return k, (lo if lo is not None else min(1, hi)), hi
    return None


class Spec:
    def __init__(self, problem: str, starter: str):
        self.args = signature(starter)
        self.alias: dict[str, str] = {}          # name -> key
        self.bounds: dict[str, list] = {}         # key -> [(kind, expr)], kind in lo/hi
        self.preds: list[str] = []                # scalar predicates over V
        self.row_preds: dict[str, list[str]] = {} # matrix -> predicates over row r
        self.alphabet: dict[str, str] = {}        # str/List[str] arg -> chars
        self.choices: dict[str, list] = {}        # element choices
        self.flags: dict[str, set] = {a: set() for a in self.args}
        self.perm_start: dict[str, int] = {}
        self.sum_cap: dict[str, str] = {}         # List[str] arg -> expr capping total length
        self.diag: dict[str, int] = {}            # matrix -> value on its diagonal
        self.zip_distinct: list[str] = []         # parallel lists whose tuples are distinct
        self.eq: list[tuple[str, str]] = []
        self.rep: dict[str, str] = {}
        _, _, cons = problem.rpartition("Constraints:")
        if not cons.strip():
            raise Unsupported("no constraints")
        text = fix_numbers(problem)
        for m in re.finditer(r"(\w+)\[i\]\s*==?\s*\[([a-z]\w*(?:,\s*[a-z]\w*)+)\]", text):
            self._row_alias(m.group(1), m.group(2).split(","))
        lines = [fix_numbers(l) for l in cons.splitlines()]
        lines = [l for l in lines if l and not l.lower().startswith(("follow", "note"))]
        self._positional_names(lines)
        for line in lines:
            self._line(line)
        self._finish()

    # --- quantities ---

    def _row_alias(self, arr, names):
        if self.args.get(arr) != "List[List[int]]":
            return
        names = [n.strip() for n in names]
        for i, n in enumerate(names):
            self.alias.setdefault(n, f"pos:{arr}:{i}")
        self.bounds.setdefault(f"inner:{arr}", []).extend([("lo", str(len(names))), ("hi", str(len(names)))])

    def _positional_names(self, lines):
        """`0 <= ai, bi < n` with one matrix argument: ai, bi are its row positions, in order."""
        mats = [a for a, t in self.args.items() if t == "List[List[int]]"
                and not any(v.startswith(f"pos:{a}:") for v in self.alias.values())]
        names = []
        for l in lines:
            for n in re.findall(r"(?<![\w\[.])([A-Za-z]i)(?![\w\[(])", l):
                if n not in self.args and n not in self.alias and n not in names:
                    names.append(n)
        if names and len(mats) == 1:
            for i, n in enumerate(names):
                self.alias[n] = f"pos:{mats[0]}:{i}"

    def key(self, t: str) -> str | None:
        """Key for an atom that names one quantity, else None."""
        t = t.strip()
        if t in self.alias:
            return self.alias[t]
        if t in self.args:
            return f"int:{t}" if self.args[t] == "int" else None
        if re.fullmatch(r"[A-Z]", t) and t.lower() in self.args:
            return self.key(t.lower())  # "1 <= K <= 10" for argument k
        m = re.fullmatch(r"(\w+)\.length", t)
        if m and m.group(1) in self.args:
            return f"len:{m.group(1)}"
        m = re.fullmatch(r"(\w+)\[[a-z]\]\.length", t)
        if m and self.args.get(m.group(1)) in ("List[List[int]]", "List[str]"):
            return f"inner:{m.group(1)}"
        m = re.fullmatch(r"(\w+)\[[a-z]\](?:\[([a-z]|\d)\])?", t)
        if m and m.group(1) in self.args:
            a, ty, sub = m.group(1), self.args[m.group(1)], m.group(2)
            if ty == "List[int]" and sub is None:
                return f"elem:{a}"
            if ty == "List[List[int]]" and sub is not None:
                return f"pos:{a}:{sub}" if sub.isdigit() else f"elem:{a}"
            if ty == "str" and sub is None or ty == "List[str]" and sub is not None and not sub.isdigit():
                return f"char:{a}"
        if re.fullmatch(r"[a-z]\w*", t) and t not in ("min", "max", "floor"):
            return self.alias.setdefault(t, f"free:{t}")  # n, m, q: a named size
        return None

    def _row_of(self, t: str) -> str | None:
        for m in re.finditer(r"\w+(?:\[\w+\])*(?:\.length)?", t):
            k = self.key(m.group(0)) if not m.group(0).isdigit() else None
            if k and k.startswith("pos:"):
                return k.split(":")[1]
        return None

    def expr(self, t: str, row: str | None = None) -> str:
        """Python source for an arithmetic atom over scalars V (and row r of `row`)."""
        t = t.strip()
        out, pos = [], 0
        for m in re.finditer(r"\w+(?:\[\w+\])*(?:\.length)?", t):
            tok = m.group(0)
            out.append(t[pos:m.start()])
            if tok.isdigit():
                out.append(tok)
            elif tok in ("min", "max", "floor") and t[m.end():].lstrip().startswith("("):
                out.append("" if tok == "floor" else tok)
            else:
                k = self.key(tok)
                if k is None:
                    raise Unsupported(f"atom {tok!r} in {t!r}")
                out.append(self._ref(k, row))
            pos = m.end()
        out.append(t[pos:])
        src = "".join(out).replace("/", "//").replace("^", "**")
        if re.search(r"[^\w\s\[\]\"'()+\-*/%:,]", src):
            raise Unsupported(f"expr {t!r}")
        return src

    @staticmethod
    def _ref(k, row):
        if k.startswith("pos:"):
            arr, i = k.split(":")[1:]
            if row != arr:
                raise Unsupported(f"position {k} outside its row")
            return f"r[{i}]"
        if k.startswith(("elem:", "char:")):
            raise Unsupported(f"element {k} in arithmetic")
        return f'V["{k}"]'

    # --- lines ---

    def _line(self, line: str):
        low = line.lower()
        line = re.sub(r"\s+for (?:all )?i\s*>=?\s*\d+$", "", line)
        names = [a for a in self.args if re.search(rf"\b{a}\b", line)]
        if re.fullmatch(r"(\w+)\[i\]\s*==?\s*\[.*\]", line):
            return  # row alias, read from the full text already
        if "valid tree" in low and names and all(self.args[a] in ("List[int]", "List[List[int]]") for a in names):
            for a in names:
                self.flags[a].add("tree")
            return
        m = re.fullmatch(r"(\w+)\[0\]\s*==\s*-1", line)
        if m and self.args.get(m.group(1)) == "List[int]":
            return self.flags[m.group(1)].add("root")  # parent array; needs a tree line too
        m = re.fullmatch(r"(\w+)\[i\]\[i\]\s*==\s*(-?\d+)", line)
        if m and self.args.get(m.group(1)) == "List[List[int]]":
            self.diag[m.group(1)] = int(m.group(2))
            return
        if re.search(r"\b(distinct|unique|no repeated|no duplicate)", low) and \
                not re.search(r"guarantee|answer|exactly one", low):
            if len(names) == 1:
                return self.flags[names[0]].add("distinct")
            if not names and "points" in low:
                self.zip_distinct = [a for a, t in self.args.items() if t == "List[int]"]
                if len(self.zip_distinct) >= 2:
                    return
        if "permutation" in low and len(names) == 1:
            m = re.search(r"\[\s*(-?\d+)\s*,", line)
            self.perm_start[names[0]] = int(m.group(1)) if m else 0
            return self.flags[names[0]].add("perm")
        if re.search(r"sorted|non-decreasing|ascending|increasing order", low) and len(names) == 1:
            if "strictly" in low:
                self.flags[names[0]].add("distinct")
            return self.flags[names[0]].add("sorted")
        if re.fullmatch(r"\w+\[i\] is a prime( number)?", line) and len(names) == 1:
            return self.flags[names[0]].add("prime")
        if re.search(r"(does not contain|has no) leading zeros?", low) and len(names) == 1 \
                and self.args[names[0]] == "str":
            return self.flags[names[0]].add("nolead")
        m = re.fullmatch(r"(?:the input is generated such that )?(?:the )?sum(?:\(| of )(\w+)\[i\]\.length\)?\s*"
                         r"(?:<=|is (?:smaller|less) than or equal(?: to)?|is at most|does not exceed)\s*(.+)", line, re.I)
        if m and self.args.get(m.group(1)) == "List[str]":
            self.sum_cap[m.group(1)] = self.expr(m.group(2))
            return
        m = re.fullmatch(r"(.+?) is divisible by (.+)|(.+?) is a multiple of (.+)", line)
        if m:
            a, b = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            self.preds.append(f"({self.expr(a)}) % ({self.expr(b)}) == 0")
            return
        m = re.fullmatch(r"(\w+) is (even|odd)", line)
        if m:
            self.preds.append(f"({self.expr(m.group(1))}) % 2 == {int(m.group(2) == 'odd')}")
            return
        if re.search(r"\bconsists?\b|\bcontains? only\b|\b(is|are) either\b|\bis one of\b", low):
            return self._alphabet(line)
        if re.search(r"<=|>=|==|!=|<|>", line):
            return self._chain(line)
        raise Unsupported(f"line {line!r}")

    def _alphabet(self, line):
        m = re.search(r"\b(consists?(?: only)? of|contains? only|is either|are either|is one of)\b(.*)", line)
        subj, rest = line[:m.start()], m.group(2)
        subjects = [re.sub(r"^(all|each|the)\s+", "", s.strip(), flags=re.I)
                    for s in re.split(r",|\band\b", subj) if s.strip()]
        lits = [a or b for a, b in re.findall(r"'([^']*)'|\"([^\"]*)\"", rest)]
        nums = re.findall(r"(?<![\w'\"])(-?\d+)(?![\w'\"])", re.sub(r"'[^']*'|\"[^\"]*\"", "", rest))
        chars = next((cs for p, cs in LETTERS if re.search(p, rest, re.I)), "")
        if re.search(r"\bdigits?\b", rest, re.I):
            chars += "0123456789"
        for s in subjects:
            a = re.match(r"(\w+)", s).group(1)
            ty = self.args.get(a)
            if ty is None:
                raise Unsupported(f"alphabet subject {s!r}")
            if ty == "List[str]" and lits and any(len(x) > 1 for x in lits) and "[i]" not in s:
                self.choices[a] = lits
            elif ty in ("str", "List[str]"):
                extra = "".join(x for x in lits if len(x) == 1) + "".join(n for n in nums if len(n) == 1)
                cs = chars + extra
                if not cs or any(len(x) > 1 for x in lits):
                    raise Unsupported(f"alphabet {line!r}")
                self.alphabet[a] = "".join(dict.fromkeys(cs))
            elif ty in ("List[int]", "List[List[int]]") and nums and not chars and not lits:
                self.choices[a] = [int(n) for n in nums]
            else:
                raise Unsupported(f"alphabet {line!r}")

    def _chain(self, line):
        parts = re.split(r"(<=|>=|==|!=|<|>)", line)
        segs, ops = [self._split_list(s) for s in parts[0::2]], parts[1::2]
        for i, op in enumerate(ops):
            for a in segs[i]:
                for b in segs[i + 1]:
                    self._rel(a, op, b)

    @staticmethod
    def _split_list(s):
        out, depth, cur = [], 0, ""
        for ch in s:
            depth += (ch in "([") - (ch in ")]")
            if ch == "," and depth == 0:
                out.append(cur)
                cur = ""
            else:
                cur += ch
        return [x.strip() for x in out + [cur] if x.strip()]

    def _rel(self, a, op, b):
        m = re.fullmatch(r"(\d+)\s*\*\s*(\w+)", a.strip()) or re.fullmatch(r"(\d+)\s*\*\s*(\w+)", b.strip())
        if m and op in CMP and self.key(m.group(2)) and not self.key(m.group(2)).startswith(LOCAL):
            return self._scaled(int(m.group(1)), m.group(2), op, b if m.string == a.strip() else a, m.string == a.strip())
        ka, kb = self.key(a), self.key(b)
        row = self._row_of(a) or self._row_of(b)
        preds = self.row_preds.setdefault(row, []) if row else self.preds
        if op == "!=" or op == "==" and row:
            preds.append(f"({self.expr(a, row)}) {op} ({self.expr(b, row)})")
            return
        if op == "==":
            if ka and kb and not ka.startswith(LOCAL) and not kb.startswith(LOCAL):
                self.eq.append((ka, kb))
            elif ka or kb:
                k, e = (ka, b) if ka else (kb, a)
                ex = self.expr(e)
                self.bounds.setdefault(k, []).extend([("lo", ex), ("hi", ex)])
            else:
                raise Unsupported(f"{a} == {b}")
            return
        strict, d = CMP[op]
        lo_side, hi_side = (a, b) if d == 1 else (b, a)  # lo_side < or <= hi_side
        klo, khi = (ka, kb) if d == 1 else (kb, ka)
        per_item = LOCAL + ("inner:",)
        lo_loc, hi_loc = bool(klo and klo.startswith(per_item)), bool(khi and khi.startswith(per_item))
        # an element bounded by a size bounds only the element, never the size
        if row:
            bound_hi, bound_lo = bool(khi) and hi_loc, bool(klo) and lo_loc
        else:
            bound_hi, bound_lo = bool(khi) and (hi_loc or not lo_loc), bool(klo) and (lo_loc or not hi_loc)
        if bound_hi:
            self.bounds.setdefault(khi, []).append(("lo", f"({self.expr(lo_side, row)}) + {strict}"))
        if bound_lo:
            self.bounds.setdefault(klo, []).append(("hi", f"({self.expr(hi_side, row)}) - {strict}"))
        if not bound_hi and not bound_lo:
            preds.append(f"({self.expr(lo_side, row)}) {'<' if strict else '<='} ({self.expr(hi_side, row)})")

    def _scaled(self, c, atom, op, other, atom_left):
        """c * x op other (or other op c * x): a bound on x itself."""
        strict, d = CMP[op]
        k, ex = self.key(atom), self.expr(other)
        x_is_lower = atom_left == (d == 1)  # c * x <= other: x gets an upper bound
        if x_is_lower:
            self.bounds.setdefault(k, []).append(("hi", f"(({ex}) - {strict}) // {c}"))
        else:
            self.bounds.setdefault(k, []).append(("lo", f"-((-(({ex}) + {strict})) // {c})"))
        ko = self.key(other)
        if ko and not ko.startswith(LOCAL):  # the other side is a quantity too: bound it by c * x
            kind, sign = ("lo", "+") if x_is_lower else ("hi", "-")
            self.bounds.setdefault(ko, []).append((kind, f"{c} * ({self._ref(k, None)}) {sign} {strict}"))

    def _finish(self):
        for x, y in self.eq:
            rx, ry = self.find(x), self.find(y)
            if rx != ry:
                self.rep[ry] = rx
        merged = {}
        for k, bs in self.bounds.items():
            merged.setdefault(k if k.startswith(LOCAL) else self.find(k), []).extend(bs)
        self.bounds = merged
        for a, t in self.args.items():
            fl = self.flags[a]
            if "root" in fl and "tree" not in fl:
                raise Unsupported(f"{a}[0] == -1 without a tree")
            if t != "int" and self.find(f"len:{a}") not in self.bounds and "tree" not in fl:
                raise Unsupported(f"no length bound for {a}")
            if t == "int" and self.find(f"int:{a}") not in self.bounds:
                raise Unsupported(f"no bound for {a}")
            if t == "List[int]" and f"elem:{a}" not in self.bounds and a not in self.choices \
                    and not fl & {"perm", "tree"}:
                raise Unsupported(f"no value range for {a}")
            if t in ("str", "List[str]") and a not in self.alphabet and a not in self.choices:
                raise Unsupported(f"no alphabet for {a}")
            if t == "List[List[int]]":
                if self.find(f"inner:{a}") not in self.bounds and not self._positions(a):
                    raise Unsupported(f"no row length for {a}")
                if f"elem:{a}" not in self.bounds and a not in self.choices and not self._positions(a):
                    raise Unsupported(f"no value range for {a}")
            if t == "List[str]" and a not in self.choices and self.find(f"inner:{a}") not in self.bounds:
                raise Unsupported(f"no word length for {a}")
            if "prime" in fl and t != "List[int]":
                raise Unsupported(f"prime {a}")

    def find(self, k):
        while self.rep.get(k, k) != k:
            k = self.rep[k]
        return k

    def _positions(self, a):
        return sorted(int(k.split(":")[2]) for k in self.bounds if k.startswith(f"pos:{a}:"))

    # --- sampling ---

    def _scalars(self):
        ks = {f"int:{a}" if t == "int" else f"len:{a}" for a, t in self.args.items()}
        ks |= {f"inner:{a}" for a in self.args if self._rect(a)}
        ks |= {k for k in self.bounds if not k.startswith(LOCAL + ("inner:",))}
        ks |= {v for v in self.alias.values() if v.startswith("free:")}
        return {self.find(k) for k in ks}

    def _rect(self, a):
        """Rows share one length when the row-length key is tied to another name."""
        ik = f"inner:{a}"
        return self.find(ik) != ik or any(v == ik for v in self.alias.values()) or \
            any(ik in (x, y) for x, y in self.eq)

    def _val(self, V, src, row=None):
        return eval(src, {"__builtins__": {}, "min": min, "max": max},
                    {"V": _Lookup(V, self.find), "r": row})

    def _range(self, key, V, row=None):
        """(lo, hi, complete): bounds that can be evaluated now, and whether all could."""
        los, his, complete = [], [], True
        for kind, src in self.bounds.get(key, []):
            try:
                v = self._val(V, src, row)
            except (KeyError, TypeError):
                complete = False
                continue
            if v is None:
                complete = False
                continue
            (los if kind == "lo" else his).append(v)
        return (max(los) if los else None), (min(his) if his else None), complete

    def _holds(self, key, v, V, row=None):
        lo, hi, complete = self._range(key, V, row)
        return complete and (lo is None or lo <= v) and (hi is None or v <= hi)

    def sample(self, rng: random.Random, mode: str, tries: int = 300) -> dict:
        for t in range(tries):
            try:
                V = self._sample_scalars(rng, mode, t)
                if all(self._val(V, p) for p in self.preds):
                    return self._build(rng, mode, V)
            except _Retry:
                continue
        raise Unsupported(f"could not satisfy constraints in mode {mode}")

    def _sample_scalars(self, rng, mode, attempt):
        V, todo = {}, self._scalars()
        while todo:
            pick = _pick([(k, *self._range(k, V)) for k in todo])
            if pick is None:
                raise Unsupported(f"unbounded scalars {sorted(todo)}")
            k, lo, hi = pick
            if k.startswith(("len:", "inner:", "free:")):
                hi = min(hi, self._cap(k))
                if lo > hi:
                    raise _Retry
                # max mode starts at the largest size and shrinks on retries
                v = max(lo, int(hi * 0.85 ** attempt)) if mode == "max" else rng.randint(lo, min(hi, lo + 7))
            else:
                if lo > hi:
                    raise _Retry
                v = rng.choice([lo, hi, rng.randint(lo, hi)]) if mode == "max" else rng.randint(lo, min(hi, lo + 12))
            V[k] = v
            todo.discard(k)
        if not all(self._holds(k, v, V) for k, v in V.items()):
            raise _Retry
        return V

    def _cap(self, k):
        kind, _, name = k.partition(":")
        t = self.args.get(name, "")
        return STR_CAP if t == "str" or kind == "inner" and t == "List[str]" else LIST_CAP

    def _build(self, rng, mode, V):
        out = {}
        for a, t in self.args.items():
            if t == "int":
                out[a] = V[self.find(f"int:{a}")]
            elif t == "List[int]" and "tree" in self.flags[a]:
                out[a] = self._parents(rng, V[self.find(f"len:{a}")])
            elif "tree" in self.flags[a]:
                out[a] = self._tree(rng, a, V)
            else:
                n = V[self.find(f"len:{a}")]
                if t == "str":
                    out[a] = self._str(rng, a, n, mode)
                elif t == "List[int]":
                    out[a] = self._ints(rng, a, n, V, mode)
                elif t == "List[str]":
                    out[a] = self._words(rng, a, n, V, mode)
                else:
                    out[a] = self._matrix(rng, a, n, V, mode)
        if self.zip_distinct and len(set(zip(*(out[a] for a in self.zip_distinct)))) < len(out[self.zip_distinct[0]]):
            raise _Retry
        return out

    def _alpha(self, rng, a, mode):
        al = self.alphabet[a]
        return al if mode == "max" or len(al) <= 3 or rng.random() < 0.5 else al[:rng.randint(2, 3)]

    def _str(self, rng, a, n, mode):
        al = self._alpha(rng, a, mode)
        s = "".join(rng.choice(al) for _ in range(n))
        if "nolead" in self.flags[a] and n > 1 and s[0] == "0":
            nz = [c for c in self.alphabet[a] if c != "0"]
            if not nz:
                raise _Retry
            s = rng.choice(nz) + s[1:]
        return s

    def _ints(self, rng, a, n, V, mode):
        if "perm" in self.flags[a]:
            xs = list(range(self.perm_start[a], self.perm_start[a] + n))
            rng.shuffle(xs)
            return xs
        if a in self.choices:
            xs = [rng.choice(self.choices[a]) for _ in range(n)]
        else:
            lo, hi, _ = self._range(f"elem:{a}", V)
            if lo is None or hi is None or lo > hi:
                raise _Retry
            if "prime" in self.flags[a]:
                if hi > PRIME_CAP:
                    xs = [_random_prime(rng, lo, hi) for _ in range(n)]
                    return sorted(xs) if "sorted" in self.flags[a] else xs
                pool = [p for p in primes_upto(hi) if p >= lo]
                if mode == "small" and rng.random() < 0.6:
                    pool = pool[:10]
                if not pool:
                    raise _Retry
                xs = [rng.choice(pool) for _ in range(n)]
            elif "distinct" in self.flags[a]:
                if mode == "small" and rng.random() < 0.6 and hi - lo + 1 > 2 * n:
                    hi = lo + 2 * n
                if hi - lo + 1 < n:
                    raise _Retry
                xs = rng.sample(range(lo, hi + 1), n)
            else:
                if mode == "small" and rng.random() < 0.6:
                    hi = min(hi, lo + 9)
                xs = [rng.randint(lo, hi) for _ in range(n)]
        if "sorted" in self.flags[a]:
            xs.sort()
        return xs

    def _words(self, rng, a, n, V, mode):
        if a in self.choices:
            return [rng.choice(self.choices[a]) for _ in range(n)]
        ik = self.find(f"inner:{a}")
        al = self._alpha(rng, a, mode)
        lo, hi, _ = self._range(ik, V)
        if ik in V:
            lo = hi = V[ik]
        if lo is None or hi is None or lo > hi:
            raise _Retry
        hi = min(hi, STR_CAP)
        cap = min(self._val(V, self.sum_cap[a]) if a in self.sum_cap else STR_CAP, STR_CAP)
        if mode == "max":
            lens = [max(lo, min(hi, cap // max(n, 1)))] * n
        else:
            lens = [rng.randint(lo, min(hi, lo + 5)) for _ in range(n)]
        if sum(lens) > cap:
            raise _Retry
        words = ["".join(rng.choice(al) for _ in range(l)) for l in lens]
        if "distinct" in self.flags[a] and len(set(words)) < len(words):
            raise _Retry
        return words

    def _pkey(self, a, i):
        k = f"pos:{a}:{i}"
        return k if k in self.bounds else f"elem:{a}"

    def _row(self, rng, a, width, V, mode):
        if not self._positions(a) and not self.row_preds.get(a):  # i.i.d. elements
            if a in self.choices:
                return [rng.choice(self.choices[a]) for _ in range(width)]
            lo, hi, _ = self._range(f"elem:{a}", V)
            if lo is None or hi is None or lo > hi:
                raise _Retry
            if mode == "small" and rng.random() < 0.6:
                hi = min(hi, lo + 9)
            return [rng.randint(lo, hi) for _ in range(width)]
        for _ in range(100):
            r = [None] * width
            todo = set(range(width))
            while todo:
                pick = _pick([(i, *self._range(self._pkey(a, i), V, r)) for i in todo])
                if pick is None:
                    if a not in self.choices:
                        raise Unsupported(f"unbounded positions in {a}")
                    for i in todo:
                        r[i] = rng.choice(self.choices[a])
                    break
                i, lo, hi = pick
                if lo > hi:
                    break
                if mode == "small" and rng.random() < 0.6:
                    hi = min(hi, lo + 9)
                r[i] = rng.randint(lo, hi)
                todo.discard(i)
            if None not in r and all(self._holds(self._pkey(a, i), v, V, r) for i, v in enumerate(r)
                                     if self._pkey(a, i) in self.bounds) \
                    and all(self._val(V, p, r) for p in self.row_preds.get(a, [])):
                return r
        raise _Retry

    def _matrix(self, rng, a, n, V, mode):
        ik = self.find(f"inner:{a}")
        width = V.get(ik)
        if width is not None and n * width > LIST_CAP:
            raise _Retry
        rows, seen = [], set()
        for _ in range(n):
            w = width
            if w is None:
                lo, hi, _ = self._range(ik, V)
                w = len(self._positions(a)) if lo is None else \
                    hi if mode == "max" else rng.randint(lo, min(hi, lo + 5))
            for _ in range(50):
                r = self._row(rng, a, w, V, mode)
                if "distinct" not in self.flags[a] or tuple(r) not in seen:
                    break
            else:
                raise _Retry
            seen.add(tuple(r))
            rows.append(r)
        if "sorted" in self.flags[a]:
            rows.sort()
        if a in self.diag:
            for i in range(min(n, width or 0)):
                rows[i][i] = self.diag[a]
        return rows

    def _tree(self, rng, a, V):
        n = V[self.find(f"len:{a}")] + 1
        labels = list(range(n))
        rng.shuffle(labels)
        width = len(self._positions(a)) or 2
        edges = []
        for i in range(1, n):
            u, v = labels[i], labels[rng.randrange(i)]
            e = [u, v] if rng.random() < 0.5 else [v, u]
            for j in range(2, width):
                lo, hi, _ = self._range(f"pos:{a}:{j}", V, e + [None] * (width - len(e)))
                if lo is None or hi is None:
                    raise Unsupported(f"no range for {a} position {j}")
                e.append(rng.randint(lo, hi))
            edges.append(e)
        rng.shuffle(edges)
        return edges

    @staticmethod
    def _parents(rng, n):
        """Parent array of a random tree rooted at 0: parent[0] == -1."""
        order = [0] + rng.sample(range(1, n), n - 1)
        par = [-1] * n
        for i in range(1, n):
            par[order[i]] = order[rng.randrange(i)]
        return par


class _Lookup(dict):
    def __init__(self, V, find):
        super().__init__(V)
        self._find = find

    def __getitem__(self, k):
        return dict.__getitem__(self, self._find(k))


def render(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def sample_inputs(problem: str, starter: str, seed: int) -> tuple[list[str], list[str]]:
    """(small, max) argument strings; raises Unsupported when the constraints do not parse."""
    s, rng = Spec(problem, starter), random.Random(seed)
    small = list(dict.fromkeys(render(s.sample(rng, "small")) for _ in range(SMALL)))
    return small, [render(s.sample(rng, "max")) for _ in range(MAX)]


def _inputs(p: dict) -> tuple[int, list[str] | None, str | None]:
    try:
        small, big = sample_inputs(p["problem"], p["starter_code"], p["index"])
    except Unsupported as e:
        return p["index"], None, f"unsupported: {e}"
    except Exception as e:  # prose the grammar half-matched; treated like an unsupported line
        return p["index"], None, f"parse error: {e!r}"
    have = set(p["tests"])
    return p["index"], [t for t in small if t not in have] + big, None


def main() -> None:
    from pathlib import Path
    from pipeline.core.io import save_json
    from pipeline.tasks.leetcode.task import spec_of
    from pipeline.tasks.leetcode.verifier import service

    ap = argparse.ArgumentParser(description="Append generated limit tests to a corpus's problem files")
    ap.add_argument("problems_dir", help="holds primitives.json and the partition files")
    a = ap.parse_args()

    files = sorted(Path(a.problems_dir).glob("*.json"))
    problems = json.load(open(Path(a.problems_dir) / "primitives.json"))
    if any("limit_tests" in p for p in problems):
        raise SystemExit("limit tests already added")
    calls = [p for p in problems if p.get("mode", "call") == "call"]
    cpus = service.physical_cores()
    with ProcessPoolExecutor(len(cpus)) as ex:
        gen = {i: (tests, err) for i, tests, err in ex.map(_inputs, calls, chunksize=4)}

    pool = service.VerifierPool(cpus)
    added, why = {}, Counter()
    for n, p in enumerate(calls):
        tests, err = gen[p["index"]]
        if err:
            why[err.split(":")[0] + " (original tests only)"] += 1
            added[p["index"]] = ([], [])
            continue
        runs = pool.gold_outputs({**spec_of(p), "tests": tests})
        is_max = [j >= len(tests) - MAX for j in range(len(tests))]
        keep = [(t, g["expected"], m) for t, g, m in zip(tests, runs, is_max)
                if g["expected"] is not None and not (m and g["cpu_s"] > MAX_GOLD_S)]
        added[p["index"]] = ([t for t, _, _ in keep], [e for _, e, _ in keep])
        why[f"parsed, {sum(m for _, _, m in keep)}/{MAX} max inputs kept"] += 1
        if n % 200 == 0:
            print(f"  gold {n}/{len(calls)}", flush=True)
    pool.close()

    text = {p["index"]: p["problem"] for p in problems}
    for f in files:
        rows = problems if f.name == "primitives.json" else json.load(open(f))
        for r in rows:
            if r["index"] not in added:
                continue
            assert r["problem"] == text[r["index"]], f"{f.name}: index {r['index']} differs from primitives.json"
            tests, expected = added[r["index"]]
            r["tests"], r["expected"] = list(r["tests"]) + tests, list(r["expected"]) + expected
            r["limit_tests"] = len(tests)
        save_json(f, rows)
        print(f"{f}: {sum(r['index'] in added for r in rows)} problems patched", flush=True)
    for k, v in sorted(why.items()):
        print(f"  {v:>5} {k}")


if __name__ == "__main__":
    main()
