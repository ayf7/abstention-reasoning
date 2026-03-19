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

    # Check for abstention (always detect, but only assign
    # abstention_score when reward_abstain is True)
    if has_abstain_tag(solution_str):
        abs_score = abstention_score if reward_abstain else 0
        return {
            "score": abs_score,
            "score_wo_hint_penalty": abs_score,
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
            penalized_hints = min(num_hints, 2)
            final_score = base_score * (1 - hint_penalty * penalized_hints)
        else:
            final_score = base_score
    else:
        final_score = format_score
        if hint_bonus > 0 and num_hints > 0:
            penalized_hints = min(num_hints, 2)
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


def compute_score_hint_dynamic(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    format_score=0.1,
    score=1.0,
    correct_end=0.55,
    incorrect_end=0.45,
    max_hints=5,
    **kwargs,
):
    """Dynamic hint scoring where correct and incorrect converge to separate endpoints.

    correct(n)  = score - (score - correct_end) * n / max_hints              (inclusive)
    wrong(n)    = format_score + (incorrect_end - format_score) * n / (max_hints + 1)  (exclusive)
    malformed   = 0
    """
    result = compute_score(
        data_source, solution_str, ground_truth, extra_info,
        format_score=format_score, penalize_hint=False, **kwargs,
    )

    n = min(result['num_hints'], max_hints)

    if result.get('malformed', False):
        result['score'] = 0.0
    elif result['correct']:
        result['score'] = score - (score - correct_end) * n / max_hints
    else:
        result['score'] = format_score + (incorrect_end - format_score) * n / (max_hints + 1)

    return result


def compute_score_dynamic_abstain(
    data_sources, solution_strs, ground_truths, extra_infos,
    uids=None, r_c=1.0, r_w=0.1, **kwargs,
):
    """Batch reward function for adaptive abstention (used with AdaptiveRewardManager).

    Classifies each sample as correct, abstained, or wrong. Returns per-sample
    dicts with 'score', 'correct', 'abstained' fields. The AdaptiveRewardManager
    overrides the score for abstained samples with the EMA-derived r_a.

    Must be used with reward_manager: adaptive (AdaptiveRewardManager).
    """
    results = []
    for i in range(len(solution_strs)):
        extra_info = extra_infos[i] if extra_infos[i] is not None else {}
        result = compute_score(
            data_source=data_sources[i],
            solution_str=solution_strs[i],
            ground_truth=ground_truths[i],
            extra_info=extra_info,
            **kwargs,
        )
        if not result.get("abstained", False):
            if not has_malformed_structure(solution_strs[i]) and has_abstain_tag(solution_strs[i]):
                result["abstained"] = True
        if result.get("correct", False):
            result["score"] = r_c
        results.append(result)
    return results


def compute_score_damped_abstain(
    data_sources, solution_strs, ground_truths, extra_infos,
    uids=None, r_c=1.0, r_w=0.1, **kwargs,
):
    """Batch reward function with damped per-group abstention scoring.

    Computes per-group abstention reward:
        r_a(x) = r_w + (r_c - r_w) * (1 - p_hat) * (1 - n_a / G)

    where p_hat is accuracy among non-abstaining samples and n_a/G is
    the group abstention rate. Anti-collapse: as n_a -> G, r_a -> r_w.

    Must be used with reward_manager: batch (BatchRewardManager).
    """
    from collections import defaultdict

    # First pass: classify each sample using per-sample scoring
    results = []
    for i in range(len(solution_strs)):
        extra_info = extra_infos[i] if extra_infos[i] is not None else {}
        result = compute_score(
            data_source=data_sources[i],
            solution_str=solution_strs[i],
            ground_truth=ground_truths[i],
            extra_info=extra_info,
            **kwargs,
        )
        # Detect abstention (compute_score without reward_abstain=True won't set this)
        if not result.get("abstained", False):
            if not has_malformed_structure(solution_strs[i]) and has_abstain_tag(solution_strs[i]):
                result["abstained"] = True
        results.append(result)

    if uids is None:
        return results

    # Group by uid
    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[uid].append(i)

    # Second pass: compute dynamic r_a per group and assign rewards
    for uid, indices in groups.items():
        G = len(indices)
        k = sum(1 for i in indices if results[i].get("correct", False))
        n_a = sum(1 for i in indices if results[i].get("abstained", False))
        n_attempts = G - n_a

        if n_attempts > 0:
            p_hat = k / n_attempts
        else:
            p_hat = 0.0

        damping = 1.0 - (n_a / G)
        r_a_dynamic = r_w + (r_c - r_w) * (1.0 - p_hat) * damping

        for i in indices:
            if results[i].get("correct", False):
                results[i]["score"] = r_c
            elif results[i].get("abstained", False):
                results[i]["score"] = r_a_dynamic
            # else: keep original per-sample score (format_score or 0)

            results[i]["r_a_dynamic"] = r_a_dynamic
            results[i]["group_p_hat"] = p_hat
            results[i]["group_n_a"] = n_a
            results[i]["group_size"] = G

    return results


def compute_score_hint_exponential(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    format_score=0.1,
    score=1.0,
    final=0.5,
    max_hints=5,
    base=0.5,
    **kwargs,
):
    """Exponential hint scoring where early hints are expensive and later hints are cheap.

    Uses exponential interpolation factor: (1 - base^n) / (1 - base^max_hints)

    correct(n)  = score - (score - final) * factor(n)
    wrong(n)    = format_score + (final - format_score) * factor_wrong(n)
    malformed   = 0

    At n=0: correct = score, wrong = format_score
    At n=max_hints: correct = final
    """
    result = compute_score(
        data_source, solution_str, ground_truth, extra_info,
        format_score=format_score, penalize_hint=False, **kwargs,
    )

    n = min(result['num_hints'], max_hints)

    if result.get('malformed', False):
        result['score'] = 0.0
    elif result['correct']:
        factor = (1 - base ** n) / (1 - base ** max_hints)
        result['score'] = score - (score - final) * factor
    else:
        factor = (1 - base ** n) / (1 - base ** (max_hints + 1))
        result['score'] = format_score + (final - format_score) * factor

    return result
