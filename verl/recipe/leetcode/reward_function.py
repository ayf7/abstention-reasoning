"""Reward function for the leetcode task (verl `batch` reward manager).

1.0 iff the verifier service returns PASS (all tests correct and CPU time
within the gold-calibrated gate). A finished, wrong attempt earns
format_score: reasoning closed with </think>, code in <answer>...</answer>
that compiles, and a verdict of WRONG_ANSWER, RUNTIME_ERROR or TIMEOUT.
Everything else is 0.0: reasoning cut off at the length limit, no answer,
code that does not compile, TOO_SLOW, and INFRA_ERROR.

With rollout answer forcing, a rollout whose answer was forced loses
forced_penalty from whatever it scored (so a forced failure can go negative).

method_c's verifier rows carry a 0/1 label instead of tests: 1.0 for the
right label, format_score for the wrong one, 0.0 for no label.

method_b (compute_score_hint) asks for hints inline: a response that breaks the
nested request grammar scores 0.0, a pass loses hint_penalty per hint used,
and every request made after the hints ran out costs exhausted_penalty (more
than max_exhausted_requests of them scores 0.0), as in the math recipe.

The whole step's responses go to the service in one call. The service
address comes from $VERIFIER_SOCKET, which `train_rl` sets after starting
the service.
"""

import importlib.util
import re
from pathlib import Path

from pipeline.tasks.leetcode.verifier.verifier import (
    INFRA_ERROR, MALFORMED, PASS, RUNTIME_ERROR, TIMEOUT, TOO_SLOW, WRONG_ANSWER,
)
from pipeline.tasks.leetcode.task import extract_code, is_label, predicted_label, verify_batch

ATTEMPT_VERDICTS = {WRONG_ANSWER, RUNTIME_ERROR, TIMEOUT}
# Load-stage errors the sandbox reports by exception name when compile() fails.
SYNTAX_ERRORS = {"SyntaxError", "IndentationError", "TabError"}
# What the rollout loop answers once the hints are used up (verl/verl/workers/rollout/hint_loop.py).
NO_MORE_HINTS = "No more hints available."

# The inline-request grammar, shared with the SFT filter. Loaded by path: recipes are standalone files.
_GRAMMAR_SPEC = importlib.util.spec_from_file_location(
    "nested_grammar", Path(__file__).resolve().parents[1] / "shared" / "nested_grammar.py")
_grammar = importlib.util.module_from_spec(_GRAMMAR_SPEC)
_GRAMMAR_SPEC.loader.exec_module(_grammar)


def compute_score(data_sources, solution_strs, ground_truths, extra_infos=None, uids=None,
                  format_score=0.1, forced=None, forced_penalty=0.0, **kwargs):
    if all(is_label(gt) for gt in ground_truths):
        return [_label_score(s, gt["correct"], format_score) for s, gt in zip(solution_strs, ground_truths)]
    results = verify_batch(list(ground_truths), [extract_code(s) for s in solution_strs])
    scores = []
    forced = [0.0] * len(solution_strs) if forced is None else [float(f) for f in forced]
    for s, r, f in zip(solution_strs, results, forced):
        truncated = "</think>" not in s
        attempted = (not truncated and r["verdict"] in ATTEMPT_VERDICTS
                     and r.get("error") not in SYNTAX_ERRORS)
        correct = r["verdict"] == PASS
        scores.append({
            "score": (1.0 if correct else (format_score if attempted else 0.0)) - forced_penalty * f,
            "correct": float(correct),
            "attempted": float(attempted),
            "forced": f,
            "truncated": float(truncated),
            "malformed": float(r["verdict"] == MALFORMED),
            "too_slow": float(r["verdict"] == TOO_SLOW),
            "infra_error": float(r["verdict"] == INFRA_ERROR),
        })
    return scores


def compute_score_hint(data_sources, solution_strs, ground_truths, extra_infos=None, uids=None,
                       format_score=0.1, hint_penalty=0.1, exhausted_penalty=0.0,
                       max_exhausted_requests=None, **kwargs):
    scores = compute_score(data_sources, solution_strs, ground_truths, extra_infos, uids, format_score, **kwargs)
    for s, sc in zip(solution_strs, scores):
        responses = [r.strip() for r in re.findall(r"<response>(.*?)</response>", s, re.DOTALL)]
        exhausted = responses.count(NO_MORE_HINTS)
        used = len(responses) - exhausted
        malformed = _grammar.has_malformed_structure_nested(s)
        score = 0.0 if malformed else (sc["score"] * (1 - hint_penalty * used) if sc["correct"] else sc["score"])
        if max_exhausted_requests is not None and exhausted > max_exhausted_requests:
            score = 0.0
        sc.update(score=max(score - exhausted_penalty * exhausted, 0.0), score_wo_hint_penalty=0.0 if malformed else sc["score"],
                  num_hints=float(used), num_exhausted=float(exhausted), malformed_structure=float(malformed))
    return scores


def _label_score(solution_str, label, format_score):
    pred = predicted_label(solution_str)
    correct = pred == label
    return {
        "score": 1.0 if correct else (format_score if pred is not None else 0.0),
        "correct": float(correct),
        "attempted": float(pred is not None),
        "truncated": float("</think>" not in solution_str),
        "predicted_1": float(pred == 1),
    }
