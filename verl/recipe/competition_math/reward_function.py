"""Reward functions for competition_math task."""

import re
import random


def extract_answer(solution_str: str) -> str | None:
    """Extract answer from <answer>...</answer> tags."""
    pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(pattern, solution_str, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def has_abstain_tag(solution_str: str) -> bool:
    """Check if the solution ends with the abstention pattern: </think>\\n\\n<abstain>"""
    return solution_str.rstrip().endswith("</think>\n\n<abstain>")


def check_answer(predicted: str, correct: str) -> bool:
    """Check if predicted answer matches correct answer using math-verify."""
    if not predicted or not correct:
        return False

    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

    # Primary: parse both as LaTeX (wrapped in $...$)
    try:
        gold_parsed = parse(f"${correct}$", extraction_config=[LatexExtractionConfig()])
    except Exception:
        gold_parsed = []

    try:
        pred_parsed = parse(f"${predicted}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
    except Exception:
        pred_parsed = []

    if gold_parsed and pred_parsed:
        try:
            if verify(gold_parsed, pred_parsed):
                return True
        except Exception:
            pass

    # Fallback: try both as plain expressions
    try:
        gold_plain = parse(correct, extraction_config=[ExprExtractionConfig()])
        pred_plain = parse(predicted, extraction_config=[ExprExtractionConfig()])
        if gold_plain and pred_plain:
            return verify(gold_plain, pred_plain)
    except Exception:
        pass

    return False


def has_malformed_structure(solution_str: str) -> bool:
    """Validate the overall tag structure of the response.

    The response starts inside an open <think> block (from assistant prefix).
    Valid structure:
        ([text]</think><request></request><response>...</response><think>)*
        [text]</think>\\n\\n(<answer>...</answer> | <abstain>)

    Validates:
    - Correct tag sequence (state machine)
    - <request> tags are tight (no content inside)
    - No spurious/duplicate tags (e.g. double </think>)

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

    i = 0
    while i < len(tags):
        if tags[i] != '</think>':
            return True
        i += 1

        if i >= len(tags):
            return True

        if tags[i] == '<request>':
            expected = ['<request>', '</request>', '<response>', '</response>', '<think>']
            for expected_tag in expected:
                if i >= len(tags) or tags[i] != expected_tag:
                    return True
                i += 1
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


def get_num_hints(solution_str: str) -> int:
    """Count all hint request/response exchanges (including exhausted ones)."""
    responses = re.findall(r'<response>(.*?)</response>', solution_str, re.DOTALL)
    return len(responses)


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


def compute_score(
    data_source,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    format_score: float = 0.1,
    score: float = 1.0,
    reward_abstain: bool = False,
    abstention_score: float = 0.5,
    penalize_hint: bool = False,
    hint_penalty: float = 0.1,
    hint_bonus: float = 0.0,
    nested_request: bool = False,
    **kwargs,
) -> dict:
    """
    Compute reward score for competition_math task.

    Args:
        data_source: Data source identifier
        solution_str: Model's complete response
        ground_truth: Dict with 'answer' key containing correct answer
        extra_info: Additional context
        format_score: Partial credit for well-formatted but wrong answer
        score: Full score for correct answer
        reward_abstain: Whether to reward abstention
        abstention_score: Score for abstaining
        penalize_hint: Whether to penalize hint usage
        hint_penalty: Penalty per hint used (multiplicative)
        hint_bonus: Bonus added to format_score when hints were used and
            answer is wrong but formatted (default 0.0, no bonus)
        nested_request: Score against the inline-request grammar, where
            <request> sits inside <think> (default False)

    Returns:
        Dict with score and metadata
    """
    correct_answer = ground_truth.get("answer", "")
    do_print = False

    if do_print:
        print(f"--------------------------------")
        print(f"Correct answer: {correct_answer}")
        print(f"Solution: {solution_str[:500]}...")

    # Counted before the malformed check: a truncated rollout still consumed
    # whatever hints it was given, and reporting 0 for it makes the logged
    # hint-usage rate track the malformed rate instead of actual hint use.
    num_hints = get_num_hints(solution_str)

    # Structural validation: verify entire tag sequence is well-formed.
    # nested_request methods keep the request inside <think>, a different and
    # incompatible grammar, so the validator is selected rather than patched.
    validator = (has_malformed_structure_nested if nested_request
                 else has_malformed_structure)
    if validator(solution_str):
        if do_print:
            print(f"Malformed structure detected - awarding 0")
        return {
            "score": 0,
            "score_wo_hint_penalty": 0,
            "num_hints": num_hints,
            "abstained": False,
            "correct": False,
            "malformed": True,
        }

    # Check for abstention first (always detect, but only assign
    # abstention_score when reward_abstain is True)
    if has_abstain_tag(solution_str):
        abs_score = abstention_score if reward_abstain else 0
        if do_print:
            print(f"Abstaining - awarding {abs_score}")
        return {
            "score": abs_score,
            "score_wo_hint_penalty": abs_score,
            "num_hints": num_hints,
            "abstained": True,
            "correct": False,
            "malformed": False,
        }

    # Extract predicted answer
    predicted = extract_answer(solution_str)

    if predicted is None:
        if do_print:
            print(f"No answer found")
        return {
            "score": 0,
            "score_wo_hint_penalty": 0,
            "num_hints": num_hints,
            "abstained": False,
            "correct": False,
            "malformed": True,
        }

    # Check correctness
    is_correct = check_answer(predicted, correct_answer)

    if is_correct:
        if do_print:
            print(f"Correct! Predicted: {predicted}")
        base_score = score
        if penalize_hint and num_hints > 0:
            penalized_hints = min(num_hints, 5)
            final_score = base_score * (1 - hint_penalty * penalized_hints)
            final_score = max(final_score, 0)  # Don't go negative
        else:
            final_score = base_score
        return {
            "score": final_score,
            "score_wo_hint_penalty": base_score,
            "num_hints": num_hints,
            "abstained": False,
            "correct": True,
            "malformed": False,
        }
    else:
        final_format_score = format_score
        if hint_bonus > 0 and num_hints > 0:
            penalized_hints = min(num_hints, 5)
            final_format_score = format_score + hint_bonus * penalized_hints
        if do_print:
            print(f"Wrong. Predicted: {predicted}, Expected: {correct_answer}")
            if hint_bonus > 0 and num_hints > 0:
                print(f"  Hint bonus applied: {format_score} + {hint_bonus}*{num_hints} = {final_format_score}")
        return {
            "score": final_format_score,
            "score_wo_hint_penalty": final_format_score,
            "num_hints": num_hints,
            "abstained": False,
            "correct": False,
            "malformed": False,
        }


def has_commit_tag(solution_str: str) -> bool:
    """Check if the solution ends with <commit> after <answer>...</answer>."""
    return bool(re.search(r'</answer>\s*<commit>\s*$', solution_str.rstrip()))


def has_abstain_after_answer(solution_str: str) -> bool:
    """Check if the solution ends with <abstain> after <answer>...</answer>."""
    return bool(re.search(r'</answer>\s*<abstain>\s*$', solution_str.rstrip()))


def has_malformed_structure_commit(solution_str: str) -> bool:
    """Validate tag structure for abstention_commit format.

    Valid structure:
        [text]</think>\\n\\n<answer>...</answer>\\n<commit>
        [text]</think>\\n\\n<answer>...</answer>\\n<abstain>

    Returns True if malformed, False if valid.
    """
    tag_pattern = r'(</think>|<think>|<answer>|</answer>|<commit>|<abstain>)'
    tags = re.findall(tag_pattern, solution_str)

    if not tags:
        return True

    # Expected: </think> <answer> </answer> (<commit> | <abstain>)
    if len(tags) != 4:
        return True

    if tags[0] != '</think>':
        return True
    if tags[1] != '<answer>':
        return True
    if tags[2] != '</answer>':
        return True
    if tags[3] not in ('<commit>', '<abstain>'):
        return True

    return False


def compute_score_abstention_commit(
    data_source,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    commit_correct: float = 1.0,
    commit_wrong: float = 0.0,
    abstain_correct: float = 0.0,
    abstain_wrong: float = 0.5,
    format_score: float = 0.1,
    **kwargs,
) -> dict:
    """
    Reward function for abstention_commit format.

    The model must always produce an answer, then commit or abstain.
    This enables measuring both precision and recall.

    Reward matrix:
                    Correct answer    Wrong answer
        <commit>    commit_correct    commit_wrong + format_score
        <abstain>   abstain_correct   abstain_wrong + format_score
        malformed   0                 0

    Default rewards (with format_score=0.1):
        commit + correct  = 1.0  (best: confident and right)
        commit + wrong    = 0.1  (wrong but well-formatted)
        abstain + correct = 0.1  (missed opportunity but well-formatted)
        abstain + wrong   = 0.6  (good: knew it was wrong, well-formatted)
        malformed         = 0.0  (bad format)
    """
    correct_answer = ground_truth.get("answer", "")
    do_print = random.randint(1, 64) == 1

    # Structural validation
    if has_malformed_structure_commit(solution_str):
        if do_print:
            print(f"Malformed commit structure - awarding 0")
        return {
            "score": 0,
            "correct": False,
            "committed": False,
            "abstained": False,
            "malformed": True,
            "predicted_answer": None,
        }

    # Extract answer (always present in this format)
    predicted = extract_answer(solution_str)
    if predicted is None:
        return {
            "score": 0,
            "correct": False,
            "committed": False,
            "abstained": False,
            "malformed": True,
            "predicted_answer": None,
        }

    # Check correctness
    is_correct = check_answer(predicted, correct_answer)
    committed = has_commit_tag(solution_str)
    abstained = has_abstain_after_answer(solution_str)

    # Assign reward from the matrix (format_score added for well-formatted responses)
    if committed and is_correct:
        final_score = commit_correct
    elif committed and not is_correct:
        final_score = commit_wrong + format_score
    elif abstained and is_correct:
        final_score = abstain_correct + format_score
    elif abstained and not is_correct:
        final_score = abstain_wrong + format_score
    else:
        final_score = 0  # shouldn't happen if structure is valid

    if do_print:
        action = "COMMIT" if committed else "ABSTAIN"
        correctness = "CORRECT" if is_correct else "WRONG"
        print(f"[{action}+{correctness}] score={final_score} pred={predicted} gold={correct_answer}")

    return {
        "score": final_score,
        "correct": is_correct,
        "committed": committed,
        "abstained": abstained,
        "malformed": False,
        "predicted_answer": predicted,
    }


def has_malformed_structure_verify(solution_str: str) -> bool:
    """Validate tag structure for abstention_verify format.

    Valid structure:
        [text]</think>\n\n<answer>...</answer>\n\n<verify>...</verify>\n<commit>
        [text]</think>\n\n<answer>...</answer>\n\n<verify>...</verify>\n<abstain>

    Returns True if malformed, False if valid.
    """
    tag_pattern = r'(</think>|<think>|<answer>|</answer>|<verify>|</verify>|<commit>|<abstain>)'
    tags = re.findall(tag_pattern, solution_str)

    if not tags:
        return True

    # Expected: </think> <answer> </answer> <verify> </verify> (<commit> | <abstain>)
    if len(tags) != 6:
        return True

    expected = ['</think>', '<answer>', '</answer>', '<verify>', '</verify>']
    if tags[:5] != expected:
        return True
    if tags[5] not in ('<commit>', '<abstain>'):
        return True

    return False


def compute_score_abstention_verify(
    data_source,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    commit_correct: float = 1.0,
    commit_wrong: float = 0.0,
    abstain_correct: float = 0.5,
    abstain_wrong: float = 0.4,
    format_score: float = 0.1,
    **kwargs,
) -> dict:
    """
    Reward function for abstention_verify format.

    The model produces <answer>, then <verify> to check its work, then <commit>/<abstain>.

    Reward matrix:
                    Correct answer    Wrong answer
        <commit>    commit_correct    commit_wrong + format_score
        <abstain>   abstain_correct   abstain_wrong + format_score
        malformed   0                 0
    """
    correct_answer = ground_truth.get("answer", "")
    do_print = random.randint(1, 64) == 1

    # Structural validation
    if has_malformed_structure_verify(solution_str):
        if do_print:
            print(f"Malformed verify structure - awarding 0")
        return {
            "score": 0,
            "correct": False,
            "committed": False,
            "abstained": False,
            "malformed": True,
            "predicted_answer": None,
        }

    # Extract answer
    predicted = extract_answer(solution_str)
    if predicted is None:
        return {
            "score": 0,
            "correct": False,
            "committed": False,
            "abstained": False,
            "malformed": True,
            "predicted_answer": None,
        }

    # Check correctness
    is_correct = check_answer(predicted, correct_answer)
    committed = bool(re.search(r'</verify>\s*<commit>', solution_str))
    abstained = bool(re.search(r'</verify>\s*<abstain>', solution_str))

    # Assign reward
    if committed and is_correct:
        final_score = commit_correct
    elif committed and not is_correct:
        final_score = commit_wrong + format_score
    elif abstained and is_correct:
        final_score = abstain_correct + format_score
    elif abstained and not is_correct:
        final_score = abstain_wrong + format_score
    else:
        final_score = 0

    if do_print:
        action = "COMMIT" if committed else "ABSTAIN"
        correctness = "CORRECT" if is_correct else "WRONG"
        print(f"[VERIFY {action}+{correctness}] score={final_score} pred={predicted} gold={correct_answer}")

    return {
        "score": final_score,
        "correct": is_correct,
        "committed": committed,
        "abstained": abstained,
        "malformed": False,
        "predicted_answer": predicted,
    }


def compute_score_abstain(
    data_source,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    abstention_score: float = 0.5,
    **kwargs,
) -> dict:
    """
    Reward function that gives partial credit for abstention.

    Abstaining is rewarded with abstention_score (default 0.5),
    which is better than wrong (0) but worse than correct (1).
    """
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
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    hint_penalty: float = 0.1,
    hint_bonus: float = 0.0,
    format_score: float = 0.1,
    **kwargs,
) -> dict:
    """
    Reward function that penalizes hint usage.

    Each hint used reduces the score by hint_penalty (multiplicative).
    Final score = base_score * (1 - hint_penalty * num_hints)

    If hint_bonus > 0, wrong-but-formatted answers that used hints
    get format_score + hint_bonus (encourages hint exploration).
    """
    return compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        penalize_hint=True,
        hint_penalty=hint_penalty,
        hint_bonus=hint_bonus,
        format_score=format_score,
        **kwargs,
    )


def compute_score_hint_dynamic(
    data_source,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    format_score: float = 0.1,
    score: float = 1.0,
    correct_end: float = 0.55,
    incorrect_end: float = 0.45,
    max_hints: int = 5,
    **kwargs,
) -> dict:
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
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    format_score: float = 0.1,
    score: float = 1.0,
    final: float = 0.5,
    max_hints: int = 5,
    base: float = 0.5,
    **kwargs,
) -> dict:
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


def compute_score_verify(
    data_source, solution_str, ground_truth, extra_info,
    verdict_correct=1.0, verdict_wrong=0.1, format_score=0.0, **kwargs,
):
    """Reward function for cross-model verification.

    The model receives another model's generation and judges whether it is correct.
    Expected output format: <verify>...</verify><answer>correct/incorrect</answer>

    Rewards:
        verdict matches actual correctness:  verdict_correct (1.0)
        verdict wrong but format valid:      verdict_wrong (0.1)
        malformed output:                    format_score (0.0)
    """
    do_print = random.randint(1, 64) == 1
    generation_correct = ground_truth.get("generation_correct", False)

    # Validate tag structure: expect </verify>, <answer>, </answer>
    # Note: opening <verify> is the assistant prefix and not in the model's output
    tag_pattern = r'(</verify>|<answer>|</answer>)'
    tags = re.findall(tag_pattern, solution_str)
    expected_tags = ['</verify>', '<answer>', '</answer>']

    if tags != expected_tags:
        if do_print:
            print(f"[VERIFY] Malformed structure: tags={tags}, expected={expected_tags} -> {format_score}")
        return {"score": format_score, "correct": False, "malformed": True, "verdict": None, "generation_correct": generation_correct}

    # Extract verdict
    answer_match = re.search(r'<answer>\s*(correct|incorrect)\s*</answer>', solution_str, re.IGNORECASE)
    if answer_match is None:
        if do_print:
            print(f"[VERIFY] Invalid verdict (not 'correct'/'incorrect') -> {format_score}")
        return {"score": format_score, "correct": False, "malformed": True, "verdict": None, "generation_correct": generation_correct}

    verdict = answer_match.group(1).lower()
    verdict_is_correct = (verdict == "correct") == generation_correct
    final_score = verdict_correct if verdict_is_correct else verdict_wrong

    if do_print:
        print(f"[VERIFY] verdict={verdict} actual={generation_correct} match={verdict_is_correct} -> {final_score}")

    return {
        "score": final_score,
        "correct": verdict_is_correct,
        "malformed": False,
        "verdict": verdict,
        "generation_correct": generation_correct,
    }
