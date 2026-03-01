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


def has_malformed_structure(solution_str):
    """Validate overall tag structure of the response.

    Valid structure: [text]</think>\\n\\n(<answer>...</answer> | <abstain>)
    """
    tag_pattern = r'(</think>|<think>|<answer>|</answer>|<abstain>)'
    tags = re.findall(tag_pattern, solution_str)

    if not tags:
        return True

    # Simple single-turn: expect </think> followed by <answer></answer> or <abstain>
    i = 0
    while i < len(tags):
        if tags[i] != '</think>':
            return True
        i += 1

        if i >= len(tags):
            return True

        if tags[i] == '<answer>':
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
    reward_abstain=False,
    abstention_score=0.3,
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
        reward_abstain: Whether to reward abstention
        abstention_score: Score for abstention (default 0.3)

    Returns:
        Dict with score, correct, abstained
    """
    # Structural validation
    if has_malformed_structure(solution_str):
        return {"score": 0, "correct": False, "abstained": False, "malformed": True}

    # Check for abstention
    if reward_abstain and has_abstain_tag(solution_str):
        return {
            "score": abstention_score,
            "correct": False,
            "abstained": True,
            "malformed": False,
        }

    predicted = extract_solution(solution_str)

    if predicted is None:
        return {"score": 0, "correct": False, "abstained": False, "malformed": False}

    expected = ground_truth["expected_stdout"]
    is_correct = _normalize_stdout(predicted) == _normalize_stdout(expected)

    return {
        "score": 1.0 if is_correct else 0.0,
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
