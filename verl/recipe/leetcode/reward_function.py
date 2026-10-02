"""Reward function for the leetcode task (verl `batch` reward manager).

Pass/fail: 1.0 iff the verifier service returns PASS (all tests correct and
CPU time within the gold-calibrated gate), else 0.0. The whole step's
responses go to the service in one call. The service address comes from
$VERIFIER_SOCKET, which `train_rl` sets after starting the service.
"""

from pipeline.core.verifier.outputs import INFRA_ERROR, MALFORMED, PASS, TOO_SLOW
from pipeline.tasks.leetcode.task import extract_code, verify_batch


def compute_score(data_sources, solution_strs, ground_truths, extra_infos=None, uids=None, **kwargs):
    results = verify_batch(list(ground_truths), [extract_code(s) for s in solution_strs])
    return [
        {
            "score": float(r["verdict"] == PASS),
            "correct": float(r["verdict"] == PASS),
            "malformed": float(r["verdict"] == MALFORMED),
            "too_slow": float(r["verdict"] == TOO_SLOW),
            "infra_error": float(r["verdict"] == INFRA_ERROR),
        }
        for r in results
    ]
