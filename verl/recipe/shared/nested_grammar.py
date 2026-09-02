"""The inline-request tag grammar, shared by every task that sets nested_request.

The grammar is task-agnostic -- it inspects tags, never problem content -- but it
has to be identical in three places or the setting quietly breaks: the SFT filter
that decides which teacher generations to train on, the RL reward that scores
rollouts, and the rollout loop that forces an answer on truncation. Defining it
once is what keeps them from drifting.

Loaded by path, not imported: verl recipes are standalone files handed to the
trainer as custom_reward_function.path, so there is no package to import from.

    _spec = importlib.util.spec_from_file_location(
        "nested_grammar", Path(__file__).resolve().parents[1] / "shared" / "nested_grammar.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
"""

import re


def has_malformed_structure_nested(solution_str: str) -> bool:
    """Validate the tag structure of an inline-request response.

    Used by methods with nested_request: the response starts inside an open
    <think> block (from the assistant prefix) and stays there until the very
    end, so </think> is reached exactly once, immediately before the answer.

        ([text]<request></request><response>...</response>)*
        [text]</think>(<answer>...</answer> | <abstain>)

    A <think> tag anywhere in the response is malformed: the block is never
    reopened because it is never closed. Contrast has_malformed_structure,
    where the model leaves the think block to ask and re-enters it.

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
