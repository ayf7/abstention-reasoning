import re
import random
import ast
import operator
from collections import Counter


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    """if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]"""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer

def abstain_equation(equation_str):
    """Check if the equation indicates abstention."""
    if equation_str is None:
        return False
    return equation_str.strip().lower() == "i abstain"


def has_abstain_tag(solution_str):
    """Check if the solution ends with the abstention pattern: </think>\\n\\n<abstain>"""
    return solution_str.rstrip().endswith("</think>\n\n<abstain>")


def validate_equation(equation_str, available_numbers):
    """Validate that equation uses all and only the numbers in available_numbers, with exact counts."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]

        # Cast to Python ints (numpy int64 from parquet can cause Counter mismatch)
        available_numbers = [int(n) for n in available_numbers]

        # Compare exact counts
        return Counter(numbers_in_eq) == Counter(available_numbers)
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().=\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        
        def strip_trailing_result(equation_str):
            """
            If the equation ends with '= <number>' (optionally with whitespace), remove it.
            """
            return re.sub(r'\s*=\s*\d+\s*$', '', equation_str)

        # Evaluate the equation with restricted globals and locals
        result = eval(strip_trailing_result(equation_str), {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None
    
def get_num_hints(solution_str):
    """Count all hint request/response exchanges (including exhausted ones)."""
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
        return True  # No tags at all

    # Validate tag sequence with a state machine
    # Expected: (</think> <request> </request> <response> </response> <think>)* </think> (<answer> </answer> | <abstain>)
    i = 0
    while i < len(tags):
        # Must start each cycle with </think> (closing current think block)
        if tags[i] != '</think>':
            return True
        i += 1

        if i >= len(tags):
            return True  # Ended after </think> without terminal tag

        if tags[i] == '<request>':
            # Hint exchange: <request> </request> <response> </response> <think>
            expected = ['<request>', '</request>', '<response>', '</response>', '<think>']
            for expected_tag in expected:
                if i >= len(tags) or tags[i] != expected_tag:
                    return True
                i += 1
            # Loop back to expect next </think>
        elif tags[i] == '<answer>':
            # Terminal: <answer> </answer>
            if i + 1 >= len(tags) or tags[i + 1] != '</answer>':
                return True
            i += 2
            return i != len(tags)  # Malformed if extra tags after
        elif tags[i] == '<abstain>':
            i += 1
            return i != len(tags)  # Malformed if extra tags after
        else:
            return True  # Unexpected tag after </think>

    return True  # Ran out of tags without proper termination


def compute_score(data_source, solution_str, ground_truth, extra_info, method='strict', format_score=0.1, score=1., reward_abstain=False, abstention_score=0.3, penalize_hint=False, hint_penalty=0.2, hint_bonus=0.0, **kwargs):
    """The scoring function for countdown
    """
    #format_score = 0
    target = ground_truth['target']
    numbers = ground_truth['numbers']

    equation = extract_solution(solution_str=solution_str)
    do_print = False

    """if len(solution_str.split()) < 200:
        return 0"""

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    # Structural validation: verify entire tag sequence is well-formed
    if has_malformed_structure(solution_str):
        if do_print:
            print(f"Malformed structure detected - awarding 0")
        return {"score": 0, "score_wo_hint_penalty": 0, "num_hints": 0, "abstained": False, "malformed": True, "correct": False}

    num_hints = get_num_hints(solution_str)

    # Check for abstention first (always detect, but only assign
    # abstention_score when reward_abstain is True)
    if has_abstain_tag(solution_str):
        abs_score = abstention_score if reward_abstain else 0
        if do_print:
            print(f"Abstaining (via <abstain> tag) - awarding {abs_score}")
        return {
            "score": abs_score,
            "score_wo_hint_penalty": abs_score,
            "num_hints": num_hints,
            "abstained": True,
            "malformed": False,
            "correct": False,
        }

    if equation is None:
        if do_print:
            print(f"No equation found or length too short")
        return {"score": 0, "score_wo_hint_penalty": 0, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": False}

    # Check for abstention via answer content (legacy support)
    if reward_abstain and abstain_equation(equation):
        if do_print:
            print(f"Abstaining (via answer) - awarding {abstention_score}")
        return {
            "score": abstention_score,
            "score_wo_hint_penalty": abstention_score,
            "num_hints": num_hints,
            "abstained": True,
            "malformed": False,
            "correct": False,
        }

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return {"score": format_score, "score_wo_hint_penalty": format_score, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": False}

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return {"score": format_score, "score_wo_hint_penalty": format_score, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": False}

        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            if penalize_hint:
                penalized_hints = min(num_hints, 5)
                final_score = score * (1 - hint_penalty * penalized_hints)
            else:
                final_score = score
            return {"score": final_score, "score_wo_hint_penalty": score, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": True}
        else:
            final_format_score = format_score
            if hint_bonus > 0 and num_hints > 0:
                penalized_hints = min(num_hints, 5)
                final_format_score = format_score + hint_bonus * penalized_hints
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
                if hint_bonus > 0 and num_hints > 0:
                    print(f"  Hint bonus applied: {format_score} + {hint_bonus}*{num_hints} = {final_format_score}")
            return {"score": final_format_score, "score_wo_hint_penalty": final_format_score, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": False}
    except:
        if do_print:
            print(f"Error evaluating equation")
        return {"score": format_score, "score_wo_hint_penalty": format_score, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": False}



def compute_score_abstain(data_source, solution_str, ground_truth, extra_info, method='strict', format_score=0.1, score=1., abstention_score=0.5, **kwargs):
    """The scoring function for countdown that rewards abstention.

    Args:
        abstention_score: Reward for abstaining (default 0.5, between wrong and correct).
    """
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        method=method, format_score=format_score, score=score,
        reward_abstain=True, abstention_score=abstention_score, **kwargs
    )

def compute_score_abstention_commit(
    data_source, solution_str, ground_truth, extra_info,
    method='strict', commit_correct=1.0, commit_wrong=0.0,
    abstain_correct=0.5, abstain_wrong=0.4, format_score=0.1, **kwargs,
):
    """Reward function for abstention_commit format.

    Model produces <answer>...</answer> then <commit> or <abstain>.
    Format score is added for well-formed responses (not malformed).

    Reward matrix (+ format_score for well-formed):
                    Correct         Wrong
        <commit>    commit_correct  commit_wrong + format_score
        <abstain>   abstain_correct + format_score  abstain_wrong + format_score
        malformed   0               0
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    do_print = random.randint(1, 64) == 1

    # Check for commit/abstain tags
    has_commit = bool(re.search(r'</answer>\s*<commit>', solution_str))
    has_abstain = bool(re.search(r'</answer>\s*<abstain>', solution_str))

    # Validate structure: </think> <answer> </answer> (<commit>|<abstain>)
    tag_pattern = r'(</think>|<think>|<answer>|</answer>|<commit>|<abstain>)'
    tags = re.findall(tag_pattern, solution_str)
    if len(tags) != 4 or tags != ['</think>', '<answer>', '</answer>', tags[3]] or tags[3] not in ('<commit>', '<abstain>'):
        if do_print:
            print(f"Malformed commit structure - awarding 0")
        return {
            "score": 0, "correct": False, "committed": False,
            "abstained": False, "malformed": True, "predicted_answer": None,
        }

    equation = extract_solution(solution_str)
    if equation is None:
        return {
            "score": 0, "correct": False, "committed": False,
            "abstained": False, "malformed": True, "predicted_answer": None,
        }

    # Check correctness: valid numbers AND correct result
    is_correct = False
    if validate_equation(equation, numbers):
        result = evaluate_equation(equation)
        if result is not None and abs(result - target) < 1e-6:
            is_correct = True

    committed = has_commit
    abstained = has_abstain

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
        print(f"[{action}+{correctness}] score={final_score} eq={equation} target={target}")

    return {
        "score": final_score, "correct": is_correct,
        "committed": committed, "abstained": abstained,
        "malformed": False, "predicted_answer": equation,
    }


def compute_score_abstention_verify(
    data_source, solution_str, ground_truth, extra_info,
    method='strict', commit_correct=1.0, commit_wrong=0.0,
    abstain_correct=0.5, abstain_wrong=0.4, format_score=0.1, **kwargs,
):
    """Reward function for abstention_verify format.

    Model produces <answer>, then <verify> to check, then <commit>/<abstain>.

    Reward matrix:
                    Correct         Wrong
        <commit>    commit_correct  commit_wrong + format_score
        <abstain>   abstain_correct + format_score  abstain_wrong + format_score
        malformed   0               0
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    do_print = random.randint(1, 64) == 1
    # Validate structure: </think> <answer> </answer> <verify> </verify> (<commit>|<abstain>)
    tag_pattern = r'(</think>|<think>|<answer>|</answer>|<verify>|</verify>|<commit>|<abstain>)'
    tags = re.findall(tag_pattern, solution_str)
    expected = ['</think>', '<answer>', '</answer>', '<verify>', '</verify>']
    is_malformed = len(tags) != 6 or tags[:5] != expected or tags[5] not in ('<commit>', '<abstain>')

    if is_malformed:
        if do_print:
            print(f"Malformed verify structure - awarding 0")
        return {
            "score": 0, "correct": False, "committed": False,
            "abstained": False, "malformed": True, "predicted_answer": None,
        }

    equation = extract_solution(solution_str)
    if equation is None:
        return {
            "score": 0, "correct": False, "committed": False,
            "abstained": False, "malformed": True, "predicted_answer": None,
        }

    # Check correctness
    is_correct = False
    if validate_equation(equation, numbers):
        result = evaluate_equation(equation)
        if result is not None and abs(result - target) < 1e-6:
            is_correct = True

    committed = bool(re.search(r'</verify>\s*<commit>', solution_str))
    abstained = bool(re.search(r'</verify>\s*<abstain>', solution_str))

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
        print(f"[VERIFY {action}+{correctness}] score={final_score} eq={equation} target={target}")

    return {
        "score": final_score, "correct": is_correct,
        "committed": committed, "abstained": abstained,
        "malformed": False, "predicted_answer": equation,
    }


def compute_score_hint(data_source, solution_str, ground_truth, extra_info, method='strict', format_score=0.1, score=1., hint_penalty=0.1, hint_bonus=0.0, **kwargs):
    """The scoring function for countdown that penalizes hint usage.

    Args:
        hint_penalty: Multiplicative penalty per hint (default 0.1).
            Final score = accuracy * (1 - hint_penalty * num_hints)
        hint_bonus: Bonus added to format_score when hints were used and
            answer is wrong but formatted (default 0.0, no bonus).
    """
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        method=method, format_score=format_score, score=score,
        penalize_hint=True, hint_penalty=hint_penalty, hint_bonus=hint_bonus, **kwargs
    )


def compute_score_hint_dynamic(
    data_source, solution_str, ground_truth, extra_info,
    method='strict', format_score=0.1, score=1.0,
    correct_end=0.55, incorrect_end=0.45, max_hints=5,
    **kwargs,
):
    """Dynamic hint scoring where correct and incorrect converge to separate endpoints.

    correct(n)  = score - (score - correct_end) * n / max_hints              (inclusive)
    wrong(n)    = format_score + (incorrect_end - format_score) * n / (max_hints + 1)  (exclusive)
    malformed   = 0
    """
    result = compute_score(
        data_source, solution_str, ground_truth, extra_info,
        method=method, format_score=format_score, penalize_hint=False, **kwargs,
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
        # Set canonical scores: correct=r_c, wrong=keep original (format_score or 0)
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
    data_source, solution_str, ground_truth, extra_info,
    method='strict', format_score=0.1, score=1.0, final=0.5, max_hints=5,
    base=0.5, **kwargs,
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
        method=method, format_score=format_score, penalize_hint=False, **kwargs,
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