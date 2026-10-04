"""LeetCode task: implement a `Solution` method; scored by running it.

Data: LeetCodeDataset (newfacade), 2,641 train / 228 test problems. Each
record carries a preamble (imports, ListNode/TreeNode helpers), a starter
signature, a gold solution and a `check(candidate)` function of asserts.

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
import json
import os
import re
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from pipeline.core.utils import extract_answer
from pipeline.core.verifier import service
from pipeline.core.verifier.outputs import PASS
from pipeline.core.verifier.sandbox import encode
from pipeline.tasks.base import BaseTask

_REPO = "newfacade/LeetCodeDataset"
_REVISION = "215604aeed660029df7de2fea5a4d7b6ed476a08"
_FILES = {"train": "LeetCodeDataset-train.jsonl", "test": "LeetCodeDataset-test.jsonl"}

# The dataset's tests compare against one ordering, so a correct answer in
# another order would be scored wrong. Dropped until an order-free comparator
# is justified per problem.
_ANY_ORDER = re.compile(r"\bin any order\b", re.IGNORECASE)
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


def spec_of(gt: dict) -> dict:
    """Verifier problem spec from a primitive / ground_truth dict.

    Lists are rebuilt because rows read back from parquet hold numpy arrays.
    """
    return {"setup": gt["setup"], "entry": gt["entry"], "gold": gt["gold"],
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


class LeetCodeTask(BaseTask):
    name = "leetcode"

    system_message = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then provides the user with the answer."
    )
    assistant_prefix = "<think>\nLet me work through this problem step by step."

    # Ranges over the dataset's train rows; `eval` is the dataset's own test
    # split (see get_split_indices), so evaluation never touches train problems.
    SPLITS = {
        "sft_train": (0.0, 0.27),
        "sft_val": (0.27, 0.3),
        "rl_train": (0.3, 0.95),
        "rl_val": (0.95, 1.0),
        "eval": (0.0, 1.0),
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

    def create_primitives(self, num_puzzles: int | None, seed: int = 42) -> list[dict]:
        """Parse every record, then keep those whose gold passes in the sandbox.

        num_puzzles caps the train rows (test rows are always kept whole).
        """
        import random
        from huggingface_hub import hf_hub_download

        dropped = Counter()
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

        rows = self._drop_failing_gold(rows, dropped)
        for i, p in enumerate(rows):
            p["index"] = i
        by_source = Counter(p["source_split"] for p in rows)
        print(f"Kept {len(rows)} problems ({dict(by_source)}); dropped {dict(dropped)}")
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
            "variant": rec["difficulty"],
            "tags": rec["tags"],
            "source_split": source,
            "problem": rec["problem_description"].strip(),
            "starter_code": rec["starter_code"].rstrip(),
            "setup": setup,
            "entry": rec["entry_point"],
            "tests": [a for a, _ in cases],
            "expected": expected,
            "gold": rec["completion"],
        }

    @staticmethod
    def _drop_failing_gold(rows: list[dict], dropped: Counter) -> list[dict]:
        """One sandboxed gold run per problem on a local pool (own physical cores).

        Also drops golds slower than timeout_s / k: their k-times limit would be
        cut short by the wall deadline, so the gate could not be applied as specified.
        """
        keys = [service.spec_key(spec_of(r)) for r in rows]
        pool = service.VerifierPool(service.physical_cores(), gold_runs=1)
        max_gold_s = pool.timeout_s / pool.k
        try:
            gold = pool.measure_gold({k: spec_of(r) for k, r in zip(keys, rows)})
        finally:
            pool.close()
        kept = []
        for row, key in zip(rows, keys):
            g = gold[key]
            if g["error"] is not None:
                dropped[f"gold:{g['error'].split(':')[0]}"] += 1
            elif g["t"] > max_gold_s:
                dropped["gold:slow"] += 1
            else:
                kept.append(row)
        return kept

    def get_split_indices(self, total: int, split: str, seed: int = 42,
                          primitives: list[dict] | None = None) -> list[int]:
        import random

        if split not in self.SPLITS:
            raise ValueError(f"Task '{self.name}' does not define split '{split}'. Available: {list(self.SPLITS)}")
        if primitives is None:
            raise ValueError("leetcode splits follow the dataset's train/test field; pass primitives")
        source = "test" if split == "eval" else "train"
        pool = [p["index"] for p in primitives if p["source_split"] == source]
        random.Random(seed).shuffle(pool)
        lo, hi = self.SPLITS[split]
        return pool[int(len(pool) * lo):int(len(pool) * hi)]

    def get_ground_truth(self, primitive: dict) -> dict:
        return {k: primitive[k] for k in ("task_id", "variant", "setup", "entry", "tests", "expected", "gold")}

    def enrich_primitive_for_rl(self, primitive: dict) -> dict:
        # verl substitutes every primitive field into the runtime template;
        # carry only what the template uses (tests live in ground_truth).
        return {k: primitive[k] for k in ("index", "variant", "problem", "starter_code")}

    # --- prompting ---

    def format_prompt(self, primitive: dict, template: str, include_assistant_prefix: bool = True) -> list[dict]:
        content = template.replace("{problem}", primitive["problem"]).replace("{starter_code}", primitive["starter_code"])
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": content},
        ]
        if include_assistant_prefix:
            messages.append({"role": "assistant", "content": self.assistant_prefix})
        return messages

    # --- scoring ---

    def check_correctness(self, primitive: dict, generation: str) -> tuple[bool, dict]:
        return self.check_correctness_batch([primitive], [generation])[0]

    def check_correctness_batch(self, primitives: list[dict], generations: list[str]) -> list[tuple[bool, dict]]:
        codes = [extract_code(g) for g in generations]
        results = verify_batch(primitives, codes)
        out = []
        for res in results:
            meta = {k: res[k] for k in ("verdict", "error", "cpu_s", "limit_s", "tests_passed", "num_tests")}
            if res["verdict"] != PASS and meta["error"] is None:
                meta["error"] = res["verdict"].lower()
            out.append((res["verdict"] == PASS, meta))
        return out
