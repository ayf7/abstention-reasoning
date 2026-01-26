"""Shared utility functions."""

import re
from typing import Any


def tokenize_with_response_mask(
    text: str,
    tokenizer,
    response_start: str = "\n<response>",
    response_end: str = "</response>\n",
) -> tuple[list[int], list[int]]:
    """
    Tokenize text with clean boundaries around response segments.

    Splits text at response boundaries and tokenizes each segment independently,
    avoiding tokenization artifacts where characters merge across boundaries.

    Args:
        text: The full text to tokenize
        tokenizer: HuggingFace tokenizer
        response_start: Start pattern for response (default: "\\n<response>")
        response_end: End pattern for response (default: "</response>\\n")

    Returns:
        Tuple of (token_ids, response_mask) where:
        - token_ids: List of token IDs
        - response_mask: List of 0/1 where 0 = response token (mask), 1 = other (train)

    Example:
        >>> text = "thinking...\\n<response>hint</response>\\nmore thinking"
        >>> tokens, mask = tokenize_with_response_mask(text, tokenizer)
        >>> # Response segment tokens have mask=0, others have mask=1
    """
    # Build pattern that captures the full response including boundaries
    pattern = re.compile(
        f'({re.escape(response_start)}.*?{re.escape(response_end)})',
        re.DOTALL
    )

    # Split with capture group preserves matches in result
    # Result alternates: [before, response1, between, response2, ..., after]
    segments = pattern.split(text)

    all_tokens = []
    response_mask = []

    for i, segment in enumerate(segments):
        if not segment:
            continue

        # Tokenize segment independently (clean boundaries)
        tokens = tokenizer.encode(segment, add_special_tokens=False)
        all_tokens.extend(tokens)

        # Odd indices are captured response segments
        is_response = (i % 2 == 1)
        mask_value = 0 if is_response else 1
        response_mask.extend([mask_value] * len(tokens))

    return all_tokens, response_mask


def extract_answer(text: str) -> str | None:
    """Extract content between <answer> and </answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_think(text: str) -> str | None:
    """Extract content between <think> and </think> tags."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def safe_eval(expr: str) -> Any:
    """
    Safely evaluate arithmetic expressions.
    Only allows numbers and basic operators (+, -, *, /, parentheses).
    """
    # Remove whitespace
    expr = expr.strip()

    # Validate: only allow digits, operators, parentheses, spaces
    if not re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', expr):
        raise ValueError(f"Invalid expression: {expr}")

    # Evaluate with no builtins
    return eval(expr, {"__builtins__": {}}, {})


def model_short_name(model_path: str) -> str:
    """
    Extract short name from model path for filenames.

    Examples:
        "Qwen/Qwen3-14B" -> "qwen3-14b"
        "artifacts/countdown/models/sft" -> "sft"
        "/path/to/my-model" -> "my-model"
    """
    # Get last component of path
    name = model_path.rstrip("/").split("/")[-1]
    # Lowercase and clean up
    name = name.lower().replace(" ", "-")
    return name
