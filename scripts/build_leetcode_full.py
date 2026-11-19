"""Build data/leetcode_full: data/leetcode plus every LeetCode problem it lacks.

    python scripts/build_leetcode_full.py

data/leetcode is copied unchanged: same indices, splits, tests and hints. The
`leetcode_complete` source (problems missing from data/leetcode, labelled by a
doocs/leetcode solution) gets LLM tests then limit tests, and is appended to
primitives.json and eval.json with the next indices. Formatted eval prompts are
not copied, since they would lack the new problems; rebuild them with
create_prompts --data-name leetcode_full --split eval.
"""
import json
import shutil
from collections import Counter

from pipeline.core.io import save_json
from pipeline.core.method import DATA_ROOT
from pipeline.tasks.leetcode import limit_tests, llm_tests
from pipeline.tasks.leetcode.task import LeetCodeTask

SRC, DST = DATA_ROOT / "leetcode", DATA_ROOT / "leetcode_full"
REPLIES = DATA_ROOT / ".llm_test_replies" / llm_tests.MODEL
MIN_TESTS = 10  # as for TACO rows: fewer rarely separate right from wrong programs


def main():
    if DST.exists() and any(DST.iterdir()):
        raise SystemExit(f"{DST} is not empty")
    task, dropped = LeetCodeTask(), Counter()
    old = json.load(open(SRC / "problems" / "primitives.json"))
    have = {int(p["question_id"]) for p in old}
    rows = task._select_gold(task._label_with_gold(task._complete_rows(have, dropped), dropped), dropped)

    base = max(p["index"] for p in old) + 1
    # Generated tests are seeded by index, so indices are fixed before
    # generating and regenerated if a drop shifts them. LLM tests go before
    # the limit tests, which stay last (their count is `limit_tests`).
    while True:
        for i, r in enumerate(rows):
            r["index"] = base + i
        llm, llm_why = llm_tests.generate(rows, REPLIES)
        with_llm = [{**r, "tests": r["tests"] + llm[r["index"]][0], "expected": r["expected"] + llm[r["index"]][1],
                     "llm_tests": len(llm[r["index"]][0])} for r in rows]
        added, why, gold_s = limit_tests.generate(with_llm)
        enough = [len(r["tests"]) + len(added[r["index"]][0]) >= MIN_TESTS for r in with_llm]
        dropped["few_tests"] += enough.count(False)
        if all(enough):
            break
        rows = [r for r, ok in zip(rows, enough) if ok]
    rows = [limit_tests.append(r, added) for r in with_llm]
    print("llm tests:", llm_why.most_common())
    limit_tests.report(why, gold_s)
    print(f"appending {len(rows)} problems ({dict(Counter(r['variant'] for r in rows))}); dropped {dict(dropped)}")

    DST.mkdir(exist_ok=True)
    for sub in ("problems", "sft_datasets"):
        shutil.copytree(SRC / sub, DST / sub)
    shutil.copytree(SRC / "problems_with_format", DST / "problems_with_format",
                    ignore=shutil.ignore_patterns("eval__*"))
    for split in ("primitives", "eval"):
        path = DST / "problems" / f"{split}.json"
        save_json(path, (old if split == "primitives" else json.load(open(path))) + rows)
        print(f"{path}: +{len(rows)}")


if __name__ == "__main__":
    main()
