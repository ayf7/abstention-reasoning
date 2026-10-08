"""Reward function for the leetcode task (verl `batch` reward manager).

1.0 iff the verifier service returns PASS (all tests correct and CPU time
within the gold-calibrated gate). A finished, wrong attempt earns
format_score: reasoning closed with </think>, code in <answer>...</answer>
that compiles, and a verdict of WRONG_ANSWER, RUNTIME_ERROR or TIMEOUT.
Everything else is 0.0: reasoning cut off at the length limit, no answer,
code that does not compile, TOO_SLOW, and INFRA_ERROR.

The whole step's responses go to the service in one call. The service
address comes from $VERIFIER_SOCKET, which `train_rl` sets after starting
the service.
"""

from pipeline.tasks.leetcode.verifier.verifier import (
    INFRA_ERROR, MALFORMED, PASS, RUNTIME_ERROR, TIMEOUT, TOO_SLOW, WRONG_ANSWER,
)
from pipeline.tasks.leetcode.task import extract_code, verify_batch

ATTEMPT_VERDICTS = {WRONG_ANSWER, RUNTIME_ERROR, TIMEOUT}
# Load-stage errors the sandbox reports by exception name when compile() fails.
SYNTAX_ERRORS = {"SyntaxError", "IndentationError", "TabError"}


def compute_score(data_sources, solution_strs, ground_truths, extra_infos=None, uids=None,
                  format_score=0.1, **kwargs):
    results = verify_batch(list(ground_truths), [extract_code(s) for s in solution_strs])
    scores = []
    for s, r in zip(solution_strs, results):
        truncated = "</think>" not in s
        attempted = (not truncated and r["verdict"] in ATTEMPT_VERDICTS
                     and r.get("error") not in SYNTAX_ERRORS)
        correct = r["verdict"] == PASS
        scores.append({
            "score": 1.0 if correct else (format_score if attempted else 0.0),
            "correct": float(correct),
            "attempted": float(attempted),
            "truncated": float(truncated),
            "malformed": float(r["verdict"] == MALFORMED),
            "too_slow": float(r["verdict"] == TOO_SLOW),
            "infra_error": float(r["verdict"] == INFRA_ERROR),
        })
    return scores
