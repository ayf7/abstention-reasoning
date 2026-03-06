"""Reward function for code_output task.

Scores model predictions by comparing predicted stdout against expected stdout
using normalized string comparison.
"""

import re


def extract_solution(solution_str):
    """Extract the answer from <answer>...</answer> tags."""
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def has_abstain_tag(solution_str):
    """Check if the solution ends with the abstention pattern."""
    return solution_str.rstrip().endswith("</think>\n\n<abstain>")


def get_num_hints(solution_str):
    """Count all hint request/response exchanges."""
    responses = re.findall(r'<response>(.*?)</response>', solution_str, re.DOTALL)
    return len(responses)


def has_malformed_structure(solution_str):
    """Validate the overall tag structure of the response.

    The response starts inside an open <think> block (from assistant prefix).
    Valid structure:
        ([text]</think><request></request><response>...</response><think>)*
        [text]</think>\\n\\n(<answer>...</answer> | <abstain>)

    Validates:
    - Correct tag sequence (state machine)
    - <request> tags are tight (no content inside)
    - No spurious/duplicate tags

    Returns:
        True if the structure is malformed, False if valid.
    """
    # Content check: <request> tags must be tight (no content inside)
    if solution_str.count('<request>') != len(re.findall(r'<request></request>', solution_str)):
        return True

    # Extract all structural tags in order
    tag_pattern = r'(</think>|<think>|<request>|</request>|<response>|</response>|<answer>|</answer>|<abstain>)'
    tags = re.findall(tag_pattern, solution_str)

    if not tags:
        return True

    # Validate tag sequence with a state machine
    # Expected: (</think> <request> </request> <response> </response> <think>)* </think> (<answer> </answer> | <abstain>)
    i = 0
    while i < len(tags):
        if tags[i] != '</think>':
            return True
        i += 1

        if i >= len(tags):
            return True

        if tags[i] == '<request>':
            # Hint exchange: <request> </request> <response> </response> <think>
            expected = ['<request>', '</request>', '<response>', '</response>', '<think>']
            for expected_tag in expected:
                if i >= len(tags) or tags[i] != expected_tag:
                    return True
                i += 1
            # Loop back to expect next </think>
        elif tags[i] == '<answer>':
            if i + 1 >= len(tags) or tags[i + 1] != '</answer>':
                return True
            i += 2
            return i != len(tags)
        elif tags[i] == '<abstain>':
            i += 1
            return i != len(tags)
        else:
            return True

    return True


def _normalize_stdout(text):
    """Normalize stdout: strip trailing whitespace per line, strip trailing blank lines."""
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    format_score=0.1,
    reward_abstain=False,
    abstention_score=0.3,
    penalize_hint=False,
    hint_penalty=0.1,
    hint_bonus=0.0,
    **kwargs,
):
    """Score a code_output prediction.

    Compares predicted stdout against expected stdout using normalized
    string comparison.

    Args:
        data_source: Task name (unused, required by interface)
        solution_str: Model's full generation
        ground_truth: Dict with expected_stdout
        extra_info: Additional info (unused)
        format_score: Score for well-formed but wrong answer (default 0.1)
        reward_abstain: Whether to reward abstention
        abstention_score: Score for abstention (default 0.3)
        penalize_hint: Whether to penalize hint usage
        hint_penalty: Multiplicative penalty per hint (default 0.1)
        hint_bonus: Bonus added to format_score per hint when wrong (default 0.0)

    Returns:
        Dict with score, correct, abstained, malformed, num_hints
    """
    # Structural validation
    if has_malformed_structure(solution_str):
        return {"score": 0, "score_wo_hint_penalty": 0, "num_hints": 0, "correct": False, "abstained": False, "malformed": True}

    num_hints = get_num_hints(solution_str)

    # Check for abstention
    if reward_abstain and has_abstain_tag(solution_str):
        return {
            "score": abstention_score,
            "score_wo_hint_penalty": abstention_score,
            "num_hints": num_hints,
            "correct": False,
            "abstained": True,
            "malformed": False,
        }

    predicted = extract_solution(solution_str)

    if predicted is None:
        return {"score": 0, "score_wo_hint_penalty": 0, "num_hints": num_hints, "correct": False, "abstained": False, "malformed": False}

    expected = ground_truth["expected_stdout"]
    is_correct = _normalize_stdout(predicted) == _normalize_stdout(expected)

    if is_correct:
        base_score = 1.0
        if penalize_hint:
            penalized_hints = min(num_hints, 5)
            final_score = base_score * (1 - hint_penalty * penalized_hints)
        else:
            final_score = base_score
    else:
        final_score = format_score
        if hint_bonus > 0 and num_hints > 0:
            penalized_hints = min(num_hints, 5)
            final_score = format_score + hint_bonus * penalized_hints
        base_score = final_score

    return {
        "score": final_score,
        "score_wo_hint_penalty": base_score,
        "num_hints": num_hints,
        "correct": is_correct,
        "abstained": False,
        "malformed": False,
    }


def compute_score_abstain(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    abstention_score=0.5,
    **kwargs,
):
    """Score with abstention rewarded."""
    return compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        reward_abstain=True,
        abstention_score=abstention_score,
        **kwargs,
    )


def compute_score_hint(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    format_score=0.1,
    hint_penalty=0.1,
    hint_bonus=0.0,
    **kwargs,
):
    """Score with hint penalty and optional hint bonus.

    Correct answers are penalized by hint usage:
        score = 1.0 * (1 - hint_penalty * num_hints)
    Wrong but well-formed answers get format_score + hint_bonus * num_hints.
    Malformed/incomplete structure gets 0.

    Args:
        format_score: Score for well-formed but wrong answer (default 0.1)
        hint_penalty: Multiplicative penalty per hint (default 0.1).
        hint_bonus: Bonus per hint for wrong answers (default 0.0).
    """
    return compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        format_score=format_score,
        penalize_hint=True,
        hint_penalty=hint_penalty,
        hint_bonus=hint_bonus,
        **kwargs,
    )
