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
    """Check if the solution contains an <abstain> tag."""
    return "<abstain>" in solution_str


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


def has_malformed_request(solution_str):
    """Check for malformed hint request/response structure.

    Valid structure (sequential order):
        </think><request></request>
        <response>...</response>
        <think>

    Malformed if:
    - <request> has content inside (not tight <request></request>)
    - <response> appears without preceding </think><request></request>
    - Any deviation from the expected sequential pattern

    Returns:
        True if malformed structure found, False otherwise.
    """
    # First check: any <request> tag must be exactly <request></request> (tight, no content)
    open_count = solution_str.count('<request>')
    valid_request_count = len(re.findall(r'<request></request>', solution_str))
    if open_count != valid_request_count:
        return True

    # Second check: validate sequential structure around each <response>
    # Every <response> must be preceded by </think> ... <request></request> ... \n
    # Allow whitespace between tags, but structure must be: </think> then <request></request> then <response>

    # Find all <response> positions and validate what precedes each one
    response_starts = [m.start() for m in re.finditer(r'<response>', solution_str)]

    for pos in response_starts:
        # Look at what precedes this <response>
        prefix = solution_str[:pos]

        # Must have </think> followed by <request></request> followed by whitespace before <response>
        # Allow whitespace between </think> and <request>, and between </request> and <response>
        valid_prefix_pattern = r'</think>\s*<request></request>\s*$'
        if not re.search(valid_prefix_pattern, prefix):
            return True

    # Third check: every <request></request> should be followed by whitespace then <response>
    request_ends = [m.end() for m in re.finditer(r'<request></request>', solution_str)]

    for pos in request_ends:
        suffix = solution_str[pos:]
        # Must start with whitespace followed by <response> (system injects this)
        # Allow variable whitespace between </request> and <response>
        if not re.match(r'\s*<response>', suffix):
            # Exception: if this is at the very end (truncated or no more content), allow it
            if len(suffix.strip()) > 0:
                return True

    return False


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

    # Check for malformed request tags first - return 0 immediately
    if has_malformed_request(solution_str):
        if do_print:
            print(f"Malformed <request> tag detected - awarding 0")
        return {"score": 0, "score_wo_hint_penalty": 0, "num_hints": 0, "abstained": False, "malformed": True, "correct": False}

    num_hints = get_num_hints(solution_str)

    # Check for abstention first (before equation extraction check)
    if reward_abstain and has_abstain_tag(solution_str):
        if do_print:
            print(f"Abstaining (via <abstain> tag) - awarding {abstention_score}")
        return {
            "score": abstention_score,
            "score_wo_hint_penalty": abstention_score,
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