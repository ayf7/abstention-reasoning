"""LeetCode task: write Python code for a programming problem; scored by running it.

Data, chosen with `sources` (comma-separated sites):
  - "leetcode" (LeetCodeDataset, newfacade): 2,641 train / 228 test problems.
    Function-call problems ("call" mode): complete a `Solution` method. Each
    record carries a preamble (imports, ListNode/TreeNode helpers), a starter
    signature, a gold solution and a `check(candidate)` function of asserts.
    "module" mode (data/leetcode_o2) is the same but the answer is a standalone
    file with its own imports and classes; the preamble only builds test arguments.
  - "codeforces", "atcoder", "aizu" (TACO-verified, likaixin): stdin/stdout
    problems ("stdio" mode): write a whole program. Tests are input/expected
    output text pairs; the gold is the first of a few accepted solutions that
    passes in our sandbox. All rows are train rows, so `eval` stays LeetCode.
  - "leetcode_complete" (leetcode-complete, whiskwhite): the LeetCode problems
    LeetCodeDataset lacks, free and paid, up to Oct 2026. Statements are HTML
    and there are no tests, only example inputs: the tests are those inputs,
    labelled by a doocs/leetcode Python solution, which is also the gold.
    Test rows, so they land in `eval`; their own tests are too few to separate
    slow from fast solutions, so only problems whose constraints the limit-test
    generator parses are kept (it adds the tests that do).

Tests come from parsing `check` with `ast`: every assert yields the call's
argument source and an expected value, which is evaluated here (trusted) and
stored in canonical JSON. Supported assert forms:
    candidate(args) == expected
    is_same_list(candidate(args), expected)
    is_same_tree(candidate(args), expected)

A generation is correct iff the verifier service returns PASS: all tests match
and CPU time is within the gold-calibrated gate (see docs/leetcode_verifier.md).
"""

import ast
import importlib.util
import json
import os
import re
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from pipeline.core.utils import extract_answer
from pipeline.tasks.leetcode import limit_tests
from pipeline.tasks.leetcode.statement import html_to_text, repair_statement
from pipeline.tasks.leetcode.verifier import service
from pipeline.tasks.leetcode.verifier.verifier import PASS
from pipeline.tasks.leetcode.verifier.sandbox import encode
from pipeline.tasks.base import BaseTask

# The nested tag grammar the method_b reward scores with (loaded by path, as in countdown).
_GRAMMAR_SPEC = importlib.util.spec_from_file_location(
    "nested_grammar", Path(__file__).resolve().parents[3] / "verl" / "recipe" / "shared" / "nested_grammar.py")
_grammar = importlib.util.module_from_spec(_GRAMMAR_SPEC)
_GRAMMAR_SPEC.loader.exec_module(_grammar)

_REPO = "newfacade/LeetCodeDataset"
_REVISION = "215604aeed660029df7de2fea5a4d7b6ed476a08"
_FILES = {"train": "LeetCodeDataset-train.jsonl", "test": "LeetCodeDataset-test.jsonl"}

_TACO_REPO = "likaixin/TACO-verified"
_TACO_REVISION = "9e68575e01938f016199eda6f07d1a7ce96b35e1"
_TACO_FILE = "taco_verified.json"
_TACO_SOURCES = ("codeforces", "atcoder", "aizu")
_TACO_MIN_TESTS = 10  # fewer tests rarely separate right from wrong programs
_TACO_MAX_TEST_BYTES = 1 << 20  # inputs + outputs; bounds memory and gold time
_TACO_GOLD_CANDIDATES = 5

_COMPLETE_REPO = "whiskwhite/leetcode-complete"
_COMPLETE_REVISION = "34d75d56cb80e1ebc7a1b4eab6c2185e770b7682"
_COMPLETE_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl", "unsolved.jsonl")
_DOOCS_REPO = "https://github.com/doocs/leetcode"
_DOOCS_REVISION = "2104ae8bdd661a1ebb6b5c703f050b8060b51e32"
_DOOCS_CACHE = Path.home() / ".cache" / "doocs-leetcode"

# The dataset's tests compare against one ordering, so a correct answer in
# another order would be scored wrong. Dropped until an order-free comparator
# is justified per problem.
_ANY_ORDER = re.compile(r"\bin any order\b", re.IGNORECASE)
# Statements that accept several outputs, which a fixed expected output cannot
# judge, and interactive problems, which a fixed stdin cannot drive.
_MULTI_ANSWER = re.compile(
    r"print any|output any|any of them|any one of them|any such"
    r"|multiple (possible |valid |correct )?(answers|solutions)"
    r"|several (possible |valid )?(answers|solutions)"
    r"|any (valid|correct|possible|suitable|optimal) (answer|solution|one)", re.IGNORECASE)
_INTERACTIVE = re.compile(r"interactive|-----Interaction-----|\bflush\b", re.IGNORECASE)
_ANY_CASE = re.compile(r"in any case|any register|case[- ]insensitive|any letter case", re.IGNORECASE)
_SAME = {"is_same_list", "is_same_tree"}
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class _Unsupported(Exception):
    pass


def _is_candidate_call(node) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "candidate"


def parse_check(test_src: str) -> list[tuple[str, str]]:
    """[(args_src, expected_src)] from a dataset `check(candidate)` function."""
    tree = ast.parse(test_src)
    check = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check"), None)
    if check is None:
        raise _Unsupported("no_check")
    cases = []
    for stmt in check.body:
        if not isinstance(stmt, ast.Assert):
            raise _Unsupported("non_assert")
        t = stmt.test
        if (isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)
                and _is_candidate_call(t.left)):
            call, exp = t.left, t.comparators[0]
        elif (isinstance(t, ast.Call) and isinstance(t.func, ast.Name) and t.func.id in _SAME
              and len(t.args) == 2 and not t.keywords and _is_candidate_call(t.args[0])):
            call, exp = t.args
        else:
            raise _Unsupported("assert_form")
        args = [ast.unparse(a) for a in call.args] + [ast.unparse(k) for k in call.keywords]
        cases.append((", ".join(args), ast.unparse(exp)))
    if not cases:
        raise _Unsupported("no_asserts")
    return cases


def extract_code(generation: str) -> str | None:
    """Code inside <answer>...</answer>, or None.

    The answer is plain code; a fenced block inside it is also accepted (the
    last one wins), since the tags already delimit the code.
    """
    answer = extract_answer(generation)
    if not answer:
        return None
    blocks = _CODE_BLOCK.findall(answer)
    return blocks[-1] if blocks else answer


def is_label(ground_truth: dict) -> bool:
    """True for a method_a/method_c verifier row, whose ground truth is a 0/1
    label ({"correct": label}) rather than a problem's tests."""
    return "correct" in ground_truth and "tests" not in ground_truth


def predicted_label(generation: str) -> int | None:
    """The 0/1 a verifier wrote in <answer>, or None."""
    answer = extract_answer(generation)
    return int(answer) if answer in ("0", "1") else None


def canonicalize(generation: str) -> str | None:
    """Rewrite the text after </think> as "<answer>\n{code}\n</answer>", or None.

    Teachers often skip the tags and write a fenced block after thinking, or
    fence the code inside the tags. The code is the <answer> contents when the
    tags are present, else the last fenced block after </think>. None when the
    response never closed its thinking or has no code after it.
    """
    head, sep, tail = generation.partition("</think>")
    if not sep:
        return None
    code = extract_code(tail)
    if code is None:
        blocks = _CODE_BLOCK.findall(tail)
        code = blocks[-1] if blocks else None
    if not code or not code.strip():
        return None
    return f"{head}</think>\n\n<answer>\n{code.strip(chr(10)).rstrip()}\n</answer>"


def _doocs_solutions() -> dict[int, list[str]]:
    """{frontend id: Python solutions} from doocs/leetcode, Solution.py first.

    Only the Python files are checked out, at a pinned commit, into a cache.
    """
    import subprocess

    root = _DOOCS_CACHE / _DOOCS_REVISION
    if not (root / "solution").exists():
        git = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True)
        root.mkdir(parents=True, exist_ok=True)
        git("init", "-q")
        git("fetch", "-q", "--depth=1", "--filter=blob:none", _DOOCS_REPO, _DOOCS_REVISION)
        git("sparse-checkout", "set", "--no-cone", "solution/*/*/Solution*.py")
        git("checkout", "-q", "FETCH_HEAD")
    out: dict[int, list[str]] = {}
    for f in sorted((root / "solution").glob("*/*/Solution*.py"), key=lambda f: (f.parent, f.name != "Solution.py", f.name)):
        if m := re.match(r"(\d{4})\.", f.parent.name):
            out.setdefault(int(m.group(1)), []).append(f.read_text())
    return out


def spec_of(gt: dict) -> dict:
    """Verifier problem spec from a primitive / ground_truth dict.

    Lists are rebuilt because rows read back from parquet hold numpy arrays.
    Rows built before stdio support carry no mode and are function calls.
    `test_entry`, where present, wraps `entry` for checking answers; timing runs call `entry`.
    """
    return {"mode": gt.get("mode", "call"), "checker": gt.get("checker", "value"),
            "setup": gt["setup"], "entry": gt.get("test_entry", gt["entry"]), "gold": gt["gold"],
            "tests": [str(t) for t in gt["tests"]], "expected": [str(e) for e in gt["expected"]]}


def verify_batch(ground_truths: list[dict], codes: list[str | None]) -> list[dict]:
    """One service call for a batch; problems are deduplicated by spec hash."""
    problems, jobs = {}, []
    for gt, code in zip(ground_truths, codes):
        spec = spec_of(gt)
        key = service.spec_key(spec)
        problems[key] = spec
        jobs.append((key, code))
    return service.default_client().verify(problems, jobs)


def _runtime(complexity: dict) -> str:
    """The time label with its symbols spelled out: O(n^2) (where n is nums.length).

    A symbol that is itself the argument's name, such as k, needs no explanation.
    """
    label = complexity["time"]
    used = set(re.findall(r"[A-Za-z]+", label))
    terms = []
    for v in complexity["variables"]:
        symbol, quantity = v["symbol"], " ".join(v["quantity"].split())
        if symbol in used and symbol != quantity:
            terms.append(f"{symbol} is {quantity[0].lower() + quantity[1:] if ' ' in quantity else quantity}")
    return f"{label} (where {', '.join(terms)})" if terms else label


class LeetCodeTask(BaseTask):
    name = "leetcode"

    system_message = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then provides the user with the answer."
    )
    assistant_prefix = "<think>\nLet me work through this problem step by step."

    # Problem counts, not ranges (see get_split_indices). The train rows are
    # shuffled once and sliced in this order; `eval` takes the rest of them
    # plus every LeetCode test row. Sized for data/leetcode_taco; the
    # LeetCode-only pilot (data/leetcode) was partitioned with the old
    # fractional ranges and keeps its pinned partition files.
    SPLITS: dict[str, int | None] = {  # type: ignore[assignment]
        "sft_train": 1000,
        "sft_val": 250,
        "rl_train": 4688,
        "rl_val": 250,
        "eval": None,
    }

    rl_reward_manager = "batch"  # one verifier call per step, not per response

    @contextmanager
    def rl_runtime(self, env: dict):
        """Run the verifier on cores carved from this job; verl gets the rest.

        The last $VERIFIER_WORKERS (default 8) physical cores host workers;
        their hyperthread siblings are left idle so timing is not perturbed.
        """
        n = int(os.environ.get("VERIFIER_WORKERS", 8))
        cores = service.physical_cores()
        if len(cores) < n + 2:
            raise RuntimeError(f"need >= {n + 2} physical cores for {n} verifier workers + verl; have {len(cores)}")
        mine = cores[-n:]
        reserved = set().union(*(service.hyperthread_siblings(c) for c in mine))
        before = os.sched_getaffinity(0)
        address = os.path.join(tempfile.mkdtemp(prefix="verifier-"), "v.sock")
        proc = service.launch(mine, address)
        os.sched_setaffinity(0, before - reserved)  # inherited by the verl subprocess
        env[service.SOCKET_ENV] = address
        repo = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = os.pathsep.join(p for p in (repo, env.get("PYTHONPATH")) if p)
        print(f"Verifier: {n} workers on cpus {mine}; verl on {len(before - reserved)} cpus")
        try:
            yield
        finally:
            os.sched_setaffinity(0, before)
            proc.terminate()
            proc.wait()

    # --- data ---

    def create_primitives(self, num_puzzles: int | None, seed: int = 42, sources: str = "leetcode") -> list[dict]:
        """Parse every record, then keep those whose gold passes in the sandbox.

        num_puzzles caps the train rows of each dataset (LeetCode test rows
        are always kept whole). LeetCode rows come first, so a build with
        extra sources leaves their indices, and hence their splits, unchanged.
        """
        wanted = [s.strip() for s in sources.split(",") if s.strip()]
        unknown = set(wanted) - {"leetcode", "leetcode_complete", *_TACO_SOURCES}
        if unknown:
            raise ValueError(f"unknown sources {sorted(unknown)}; "
                             f"choose from leetcode, leetcode_complete, {', '.join(_TACO_SOURCES)}")

        dropped = Counter()
        rows = []
        if "leetcode" in wanted:
            rows += self._leetcode_rows(num_puzzles, seed, dropped)
        taco = [s for s in _TACO_SOURCES if s in wanted]
        if taco:
            rows += self._taco_rows(taco, num_puzzles, seed, dropped)
        if "leetcode_complete" in wanted:
            have = {int(r["question_id"]) for r in rows if r["dataset"] == "leetcode"}
            rows += self._label_with_gold(self._complete_rows(have, dropped), dropped)

        rows = self._select_gold(rows, dropped)
        for i, p in enumerate(rows):
            p["index"] = i
        by_source = Counter(f"{p['source']}/{p['source_split']}" for p in rows)
        print(f"Kept {len(rows)} problems ({dict(by_source)}); dropped {dict(dropped)}")
        return rows

    def _leetcode_rows(self, num_puzzles: int | None, seed: int, dropped: Counter) -> list[dict]:
        import random
        from huggingface_hub import hf_hub_download

        rows = []
        for source, fname in _FILES.items():
            path = hf_hub_download(_REPO, fname, repo_type="dataset", revision=_REVISION)
            with open(path) as f:
                records = [json.loads(line) for line in f]
            if source == "train" and num_puzzles is not None:
                random.Random(seed).shuffle(records)
                records = records[:num_puzzles]
            for rec in records:
                prim = self._primitive(rec, source, dropped)
                if prim is not None:
                    rows.append(prim)
        return rows

    def _primitive(self, rec: dict, source: str, dropped: Counter) -> dict | None:
        if _ANY_ORDER.search(rec["problem_description"]):
            dropped["any_order"] += 1
            return None
        setup = rec["prompt"]
        try:
            cases = parse_check(rec["test"])
            ns: dict = {}
            exec(setup, ns)
            expected = [json.dumps(encode(eval(e, ns))) for _, e in cases]
        except _Unsupported as e:
            dropped[f"unsupported:{e}"] += 1
            return None
        except Exception as e:
            dropped[f"expected_eval:{type(e).__name__}"] += 1
            return None
        return {
            "task_id": rec["task_id"],
            "question_id": rec["question_id"],
            "dataset": "leetcode",
            "source": "leetcode",
            "variant": rec["difficulty"],
            "tags": rec["tags"],
            "source_split": source,
            "problem": repair_statement(rec["question_id"], rec["problem_description"].strip()),
            "starter_code": rec["starter_code"].rstrip(),
            "mode": "call",
            "checker": "value",
            "setup": setup,
            "entry": rec["entry_point"],
            "tests": [a for a, _ in cases],
            "expected": expected,
            "gold": rec["completion"],
        }

    def _taco_rows(self, sites: list[str], num_puzzles: int | None, seed: int, dropped: Counter) -> list[dict]:
        """stdin/stdout problems from TACO-verified, streamed (the file is 1.9 GB)."""
        import random
        import ijson
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(_TACO_REPO, _TACO_FILE, repo_type="dataset", revision=_TACO_REVISION)
        rows = []
        with open(path, "rb") as f:
            for rec in ijson.items(f, "item"):
                if rec["source"] in sites:
                    prim = self._taco_primitive(rec, dropped)
                    if prim is not None:
                        rows.append(prim)
        if num_puzzles is not None:
            random.Random(seed).shuffle(rows)
            rows = sorted(rows[:num_puzzles], key=lambda p: p["question_id"])
        return rows

    @staticmethod
    def _taco_primitive(rec: dict, dropped: Counter) -> dict | None:
        io = rec["input_output"]
        io = json.loads(io) if isinstance(io, str) else io
        question = rec["question"]
        inputs, outputs = io.get("inputs", []), io.get("outputs", [])
        if io.get("fn_name"):
            reason = "taco:function_call"
        elif len(inputs) < _TACO_MIN_TESTS:
            reason = "taco:few_tests"
        elif not all(isinstance(x, str) for x in inputs + outputs):
            reason = "taco:non_text_tests"
        elif sum(map(len, inputs + outputs)) > _TACO_MAX_TEST_BYTES:
            reason = "taco:large_tests"
        elif str(rec.get("picture_num")) not in ("0", "None") or "<image>" in question:
            reason = "taco:picture"
        elif _INTERACTIVE.search(question):
            reason = "taco:interactive"
        elif _MULTI_ANSWER.search(question):
            reason = "taco:multiple_answers"
        elif not rec.get("solutions"):
            reason = "taco:no_solution"
        else:
            reason = None
        if reason:
            dropped[reason] += 1
            return None
        return {
            "task_id": f"taco-{rec['id']}",
            "question_id": rec["id"],
            "dataset": "taco",
            "source": rec["source"],
            "url": rec.get("url"),
            "variant": rec["difficulty"],
            "tags": ast.literal_eval(rec["tags"]) if isinstance(rec["tags"], str) else rec["tags"],
            "source_split": "train",
            "problem": question.strip(),
            "starter_code": "",
            "mode": "stdio",
            "checker": "tokens_anycase" if _ANY_CASE.search(question) else "tokens",
            "setup": "",
            "entry": "",
            "tests": inputs,
            "expected": [json.dumps(o) for o in outputs],
            "gold_candidates": rec["solutions"][:_TACO_GOLD_CANDIDATES],
        }

    def _complete_rows(self, have: set[int], dropped: Counter) -> list[dict]:
        """leetcode-complete problems whose frontend id is not in `have`."""
        from huggingface_hub import hf_hub_download

        # LeetCodeDataset's call preamble; of its two variants, the one that
        # also imports sortedcontainers, which the solutions may use.
        path = hf_hub_download(_REPO, _FILES["test"], repo_type="dataset", revision=_REVISION)
        with open(path) as f:
            setup = next(r["prompt"] for r in map(json.loads, f) if "SortedList" in r["prompt"])
        setup = setup.replace("from sortedcontainers import SortedList",
                              "from sortedcontainers import SortedDict, SortedList, SortedSet")
        doocs = _doocs_solutions()
        recs = {}
        for fname in _COMPLETE_FILES:
            path = hf_hub_download(_COMPLETE_REPO, fname, repo_type="dataset", revision=_COMPLETE_REVISION)
            with open(path) as f:
                for rec in map(json.loads, f):
                    if rec["frontend_id"].isdigit() and int(rec["frontend_id"]) not in have:
                        recs.setdefault(int(rec["frontend_id"]), rec)
        rows = []
        for qid, rec in sorted(recs.items()):
            prim = self._complete_primitive(rec, setup, doocs.get(qid, []), dropped)
            if prim is not None:
                rows.append(prim)
        return rows

    @staticmethod
    def _complete_primitive(rec: dict, setup: str, solutions: list[str], dropped: Counter) -> dict | None:
        starter = next((s["code"].rstrip() for s in rec["code_snippets"] if s["lang"] == "python3"), "")
        problem = html_to_text(rec["content"] or "")
        try:
            cls = ast.parse(starter + "\n        pass").body[-1]
            methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)] if isinstance(cls, ast.ClassDef) else []
        except (SyntaxError, IndexError):
            cls, methods = None, []
        if rec["category"] != "Algorithms":
            reason = "complete:not_algorithms"
        elif not solutions:
            reason = "complete:no_doocs_python"
        elif getattr(cls, "name", None) != "Solution" or len(methods) != 1:
            reason = "complete:not_one_solution_method"
        elif re.search(r"\bNode\b", starter):  # N-ary, random-pointer, graph... nodes: no builder for them
            reason = "complete:custom_node"
        elif isinstance(methods[0].returns, ast.Constant) and methods[0].returns.value is None:
            reason = "complete:in_place"  # gold returns None on every input, so any program would pass
        elif _ANY_ORDER.search(problem) or _MULTI_ANSWER.search(problem):
            reason = "complete:multiple_answers"
        else:
            reason = None
        if reason is None:
            names = [a.arg for a in methods[0].args.args[1:]]
            nodes = limit_tests.node_args(starter)
            lines = [l for l in rec["example_test_cases"].split("\n") if l.strip()]
            try:
                values = [json.loads(l) for l in lines]
                if not names or not values or len(values) % len(names):
                    raise ValueError
                cases = [dict(zip(names, values[i:i + len(names)])) for i in range(0, len(values), len(names))]
                if any(not isinstance(c[a], list) for c in cases for a in nodes):  # e.g. LCA's p, q: a node in the tree
                    raise ValueError
                tests = [limit_tests.render(c, nodes) for c in cases]
            except ValueError:
                reason = "complete:example_parse"
        if reason:
            dropped[reason] += 1
            return None
        return {
            "task_id": rec["title_slug"],
            "question_id": int(rec["frontend_id"]),
            "dataset": "leetcode_complete",
            "source": "leetcode",
            "variant": rec["difficulty"],
            "tags": rec["topic_tags"],
            "source_split": "test",
            "created_at": rec["created_at_approx"][:10],
            "problem": problem,
            "starter_code": starter,
            "mode": "call",
            "checker": "value",
            "setup": setup,
            "entry": f"Solution().{methods[0].name}",
            "tests": list(dict.fromkeys(tests)),
            "expected": None,
            "gold_candidates": solutions,
        }

    @staticmethod
    def _label_with_gold(rows: list[dict], dropped: Counter) -> list[dict]:
        """Expected outputs for rows that have only inputs, from their first solution that runs on all.

        Every other solution must agree on each test it runs; disagreement
        means the problem accepts several answers or a solution is wrong, and
        a fixed expected output cannot judge either.
        """
        if not rows:
            return rows
        pool = service.VerifierPool(service.physical_cores())
        kept = []
        try:
            for n, r in enumerate(rows):
                spec = spec_of({**r, "expected": [], "gold": None})
                outs = [[g["expected"] for g in pool.gold_outputs({**spec, "gold": c})] for c in r["gold_candidates"]]
                first = next((i for i, o in enumerate(outs) if None not in o), None)
                if first is None:
                    dropped["complete:gold_fails_example"] += 1
                elif any(o is not None and o != e for other in outs for o, e in zip(other, outs[first])):
                    dropped["complete:solutions_disagree"] += 1
                else:
                    r["expected"], r["gold_candidates"] = outs[first], [r["gold_candidates"][first]]
                    kept.append(r)
                if n % 100 == 0:
                    print(f"  labelled {n}/{len(rows)}", flush=True)
        finally:
            pool.close()
        return kept

    @staticmethod
    def _select_gold(rows: list[dict], dropped: Counter) -> list[dict]:
        """Keep problems with a gold that passes in the sandbox within the time cap.

        One sandboxed run per candidate on a local pool (own physical cores).
        LeetCode rows have one candidate (the dataset's completion); TACO rows
        try their candidates in order until one passes.
        """
        for r in rows:
            r.setdefault("gold_candidates", [r.get("gold")])
        pool = service.VerifierPool(service.physical_cores(), gold_runs=1)
        accepted, reason = set(), {}
        try:
            for attempt in range(max(len(r["gold_candidates"]) for r in rows)):
                todo = [i for i, r in enumerate(rows) if i not in accepted and attempt < len(r["gold_candidates"])]
                if not todo:
                    break
                specs = {}
                for i in todo:
                    rows[i]["gold"] = rows[i]["gold_candidates"][attempt]
                    specs[i] = spec_of(rows[i])
                gold = pool.measure_gold({service.spec_key(s): s for s in specs.values()})
                for i, s in specs.items():
                    g = gold[service.spec_key(s)]
                    if g["error"] is not None:
                        reason[i] = f"gold:{g['error'].split(':')[0]}"
                    else:
                        accepted.add(i)
                print(f"gold attempt {attempt + 1}: {len(todo)} tried, {len(accepted)} accepted so far")
        finally:
            pool.close()
        kept = []
        for i, r in enumerate(rows):
            del r["gold_candidates"]
            if i in accepted:
                kept.append(r)
            else:
                dropped[reason[i]] += 1
        return kept

    def get_split_indices(self, total: int, split: str, seed: int = 42,
                          primitives: list[dict] | None = None) -> list[int]:
        import random

        if split not in self.SPLITS:
            raise ValueError(f"Task '{self.name}' does not define split '{split}'. Available: {list(self.SPLITS)}")
        if primitives is None:
            raise ValueError("leetcode splits follow the dataset's train/test field; pass primitives")
        pool = [p["index"] for p in primitives if p["source_split"] == "train"]
        sized = sum(n for n in self.SPLITS.values() if n is not None)
        if len(pool) < sized:
            raise ValueError(f"{len(pool)} train problems cannot fill split sizes totalling {sized}")
        random.Random(seed).shuffle(pool)
        start = 0
        for name, n in self.SPLITS.items():
            if name == split:
                break
            start += n or 0
        if split == "eval":
            return pool[sized:] + [p["index"] for p in primitives if p["source_split"] == "test"]
        return pool[start:start + self.SPLITS[split]]

    def get_ground_truth(self, primitive: dict) -> dict:
        gt = {k: primitive[k] for k in ("task_id", "variant", "setup", "entry", "tests", "expected", "gold")}
        gt = {**gt, "mode": primitive.get("mode", "call"), "checker": primitive.get("checker", "value"),
              "test_entry": primitive.get("test_entry", primitive["entry"])}
        if hints := self.get_hints(primitive):
            # method_b: request k is answered with step k alone.
            gt["hint_exprs"] = [self._hint_step(hints, k) for k in range(1, len(hints) + 1)]
        return gt

    # --- prompting ---

    @staticmethod
    def _prompt_fields(primitive: dict) -> dict:
        """Template fields; the mode decides what the model is asked to write."""
        if primitive.get("mode", "call") == "stdio":
            return {
                "problem": primitive["problem"],
                "interface": "Your solution must be a complete program that reads the input from standard input "
                             "and prints the answer to standard output.",
                "answer_kind": "the complete program",
            }
        starter = f"```python\n{primitive['starter_code']}\n```"
        fields = {"problem": primitive["problem"]}
        if primitive.get("mode", "call") == "module":
            fields["interface"] = ("Your solution must be a complete Python file that includes every import and class "
                                   f"definition it uses, and it must define this starter code:\n{starter}")
            fields["answer_kind"] = "the complete Python file"
        else:
            fields["interface"] = f"Your solution must complete this starter code:\n{starter}"
            fields["answer_kind"] = "the complete Solution class"
        if "complexity" in primitive:
            fields["runtime"] = _runtime(primitive["complexity"])
        return fields

    def enrich_primitive_for_rl(self, primitive: dict) -> dict:
        # verl substitutes every primitive field into the runtime template;
        # carry only what the template uses (tests live in ground_truth).
        return {"index": primitive["index"], "variant": primitive["variant"], **self._prompt_fields(primitive)}

    def get_hints(self, primitive: dict) -> list:
        """`hint_steps`: one {insight, code, pseudocode} per level, each step holding
        one of the three. leetcode_o2: insight 1, insight 2, a gold skeleton (`# TODO`
        lines hide the rest), the gold. leetcode_o3: both insights (a list), three
        pseudocode rungs (high-level, key step in words, exact), the gold."""
        return primitive.get("hint_steps", [])

    # Header of the i-th pseudocode step, weakest first.
    PSEUDOCODE_HEADERS = (
        "Algorithm (high-level pseudocode for this problem; fill in the details and translate it into Python)",
        "Algorithm (pseudocode for this problem; the key step is spelled out line by line; translate it into Python)",
        "Algorithm (detailed pseudocode for this problem; it follows the solution step by step, translate it into Python)",
    )

    @staticmethod
    def _insights(step: dict) -> list[str]:
        """The insights a step reveals: none, one, or a list of them."""
        x = step["insight"]
        return [x] if isinstance(x, str) else x or []

    @staticmethod
    def _has_block(step: dict) -> bool:
        return bool(step["code"] or step.get("pseudocode"))

    def hint_sequence(self, hints: list, level: int) -> str:
        """Insights revealed so far, then the latest block: the skeleton or
        pseudocode, or the gold once the last step is reached."""
        if level == 0:
            return super().hint_sequence(hints, level)
        insights = [s for h in hints[:level] for s in self._insights(h)]
        text = "\n".join(f"Insight {i}: {s}" for i, s in enumerate(insights, 1))
        block_level = next((k for k in range(level, 0, -1) if self._has_block(hints[k - 1])), None)
        if block_level:
            text += "\n\n" + self._block(hints, block_level)
        return text

    def _block(self, hints: list, k: int) -> str:
        """Step k's code or pseudocode under its header."""
        step = hints[k - 1]
        if step["code"]:
            header = "Solution" if k == len(hints) else "Code so far (each `# TODO` line hides code still to be written)"
            return f"{header}:\n```python\n{step['code']}\n```"
        rung = sum(1 for h in hints[:k] if h.get("pseudocode"))
        return f"{self.PSEUDOCODE_HEADERS[rung - 1]}:\n```text\n{step['pseudocode']}\n```"

    def _hint_step(self, hints: list, k: int) -> str:
        """Step k on its own, numbered as hint_sequence numbers it."""
        if self._has_block(hints[k - 1]):
            return self._block(hints, k)
        n = sum(len(self._insights(h)) for h in hints[:k - 1])
        return "\n".join(f"Insight {n + i}: {s}" for i, s in enumerate(self._insights(hints[k - 1]), 1))

    def format_prompt(self, primitive: dict, template: str, include_assistant_prefix: bool = True) -> list[dict]:
        if "{runtime}" in template and "complexity" not in primitive:
            raise ValueError(f"Problem {primitive['index']} has no complexity label; this template needs --data-name leetcode_o2")
        content = template
        for key, value in self._prompt_fields(primitive).items():
            content = content.replace(f"{{{key}}}", value)
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": content},
        ]
        if include_assistant_prefix:
            messages.append({"role": "assistant", "content": self.assistant_prefix})
        return messages

    # --- scoring ---

    def filter_for_sft(self, examples: list[dict], include_wrong_valid_format: bool = False,
                       nested_request: bool = False) -> list[dict]:
        """Correct examples; with nested_request, only those whose requests follow the inline grammar."""
        if include_wrong_valid_format:
            raise ValueError("leetcode SFT keeps correct examples only")
        keep = [ex for ex in examples if ex.get("correct", False)]
        if nested_request:
            keep = [ex for ex in keep if not _grammar.has_malformed_structure_nested(ex.get("generation", ""))]
        return keep

    def check_correctness(self, primitive: dict, generation: str) -> tuple[bool, dict]:
        return self.check_correctness_batch([primitive], [generation])[0]

    def check_correctness_batch(self, primitives: list[dict], generations: list[str]) -> list[tuple[bool, dict]]:
        out = [None] * len(primitives)
        code_rows = []
        for i, (p, g) in enumerate(zip(primitives, generations)):
            if is_label(p):
                pred = predicted_label(g)
                out[i] = (pred == p["correct"], {"predicted_label": pred, "expected_label": p["correct"],
                                                 "error": None if pred is not None else "no_label"})
            else:
                code_rows.append(i)
        if code_rows:
            results = verify_batch([primitives[i] for i in code_rows], [extract_code(generations[i]) for i in code_rows])
            for i, res in zip(code_rows, results):
                meta = {k: res[k] for k in ("verdict", "error", "cpu_s", "limit_s", "tests_passed", "num_tests")}
                if res["verdict"] != PASS and meta["error"] is None:
                    meta["error"] = res["verdict"].lower()
                out[i] = (res["verdict"] == PASS, meta)
        return out
