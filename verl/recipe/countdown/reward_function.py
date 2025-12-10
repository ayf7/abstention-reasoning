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
    if equation_str.strip().lower() == "i abstain":
        return True


def validate_equation(equation_str, available_numbers):
    """Validate that equation uses all and only the numbers in available_numbers, with exact counts."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]

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
    
def get_num_hints(solution_str, hint_pattern="<hint>"):
    return solution_str.count(hint_pattern)


def compute_score(data_source, solution_str, ground_truth, extra_info, method='strict', format_score=0.1, score=1., reward_abstain=False, abstention_score=0.3, penalize_hint=False, hint_penalty=0.2, **kwargs):
    """The scoring function for countdown
    """
    #format_score = 0
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    """if len(solution_str.split()) < 200:
        return 0"""
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    num_hints = get_num_hints(solution_str)

    if equation is None: # or len(solution_str.split()) < 100:
        if do_print:
            print(f"No equation found or length too short")
        return {"score": 0, "score_wo_hint_penalty": 0, "num_hints": num_hints}
    
    """if reward_abstain:
        if abstain_equation(equation):
            if do_print:
               print(f"Abstaining")
            return format_score + abstention_score""" 
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return {"score": format_score, "score_wo_hint_penalty": format_score, "num_hints": num_hints}
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return {"score": format_score, "score_wo_hint_penalty": format_score, "num_hints": num_hints}
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            if penalize_hint:
                w_penalty = score * (1 - hint_penalty * num_hints)
            return {"score": w_penalty, "score_wo_hint_penalty": score, "num_hints": num_hints}
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            score = format_score  
            """if num_hints > 0:
                score = format_score + hint_penalty"""
            return {"score": format_score, "score_wo_hint_penalty": format_score, "num_hints": num_hints}
    except:
        if do_print:
            print(f"Error evaluating equation")
        return {"score": format_score, "score_wo_hint_penalty": format_score, "num_hints": num_hints}
    


def compute_score_abstain(data_source, solution_str, ground_truth, extra_info, method='strict', format_score=0.1, score=1., **kwargs):
    """The scoring function for countdown that rewards the model for abstention naively
    """

    return compute_score(data_source, solution_str, ground_truth, extra_info, method='strict', format_score=0.1, score=1., reward_abstain=True, abstention_score=0.1, **kwargs)

def compute_score_hint(data_source, solution_str, ground_truth, extra_info, method='strict', format_score=0.1, score=1., **kwargs):
    """The scoring function for countdown that rewards the model for abstention naively
    """
    return compute_score(data_source, solution_str, ground_truth, extra_info, method='strict', format_score=format_score, score=score, penalize_hint=True, hint_penalty=0.1, **kwargs)