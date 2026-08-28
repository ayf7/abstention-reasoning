"""Reward functions for competition_math_v2.

Identical scoring to competition_math; the only difference is where the hint
request lives. v1 leaves the think block to ask (`</think><request></request>
<response>...</response><think>`), v2 asks from inside it, so `</think>` is
reached exactly once, immediately before the answer.

Everything else -- answer extraction, math-verify checking, the hint penalty --
is reused verbatim from the v1 module rather than copied, so the two tasks can
never drift apart on anything but structure. `compute_score` reads
`has_malformed_structure` as a module global at call time, so rebinding it in
the loaded v1 namespace is enough to switch validators.
"""

import importlib.util
import re
from pathlib import Path

_V1_PATH = Path(__file__).resolve().parent.parent / "competition_math" / "reward_function.py"
_spec = importlib.util.spec_from_file_location("competition_math_reward_v1", _V1_PATH)
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)

extract_answer = _v1.extract_answer
check_answer = _v1.check_answer
has_abstain_tag = _v1.has_abstain_tag
get_num_hints = _v1.get_num_hints


def has_malformed_structure(solution_str: str) -> bool:
    """Validate the tag structure of an inline-request response.

    The response starts inside an open <think> block (from the assistant
    prefix) and stays there until the very end. Valid structure:

        ([text]<request></request><response>...</response>)*
        [text]</think>(<answer>...</answer> | <abstain>)

    A <think> tag anywhere in the response is malformed: the block is never
    reopened because it is never closed.

    Returns:
        True if the structure is malformed, False if valid.
    """
    # <request> tags must be tight (no content inside)
    if solution_str.count('<request>') != len(re.findall(r'<request></request>', solution_str)):
        return True

    tag_pattern = r'(</think>|<think>|<request>|</request>|<response>|</response>|<answer>|</answer>|<abstain>)'
    tags = re.findall(tag_pattern, solution_str)

    if not tags:
        return True

    i, n = 0, len(tags)
    while i < n and tags[i] == '<request>':
        for expected_tag in ('<request>', '</request>', '<response>', '</response>'):
            if i >= n or tags[i] != expected_tag:
                return True
            i += 1

    if i >= n or tags[i] != '</think>':
        return True
    i += 1

    if i >= n:
        return True

    if tags[i] == '<answer>':
        if i + 1 >= n or tags[i + 1] != '</answer>':
            return True
        i += 2
    elif tags[i] == '<abstain>':
        i += 1
    else:
        return True

    return i != n


_v1.has_malformed_structure = has_malformed_structure

compute_score = _v1.compute_score
compute_score_hint = _v1.compute_score_hint
