"""Shared utility functions."""

import re
from typing import Any


def tokenize_with_response_mask(
    text: str,
    tokenizer,
) -> tuple[list[int], list[int]]:
    """
    Tokenize text with segment boundaries matching RL rollout token splicing.

    During RL rollouts, hint injections are tokenized independently as:
        "\\n<response>hint</response>\\n<think>\\n"
    This function splits text at those same boundaries and tokenizes each
    segment independently, so SFT training sees the same token sequences
    the model will encounter during RL.

    Args:
        text: The full text to tokenize (completion portion, not prompt)
        tokenizer: HuggingFace tokenizer

    Returns:
        Tuple of (token_ids, response_mask) where:
        - token_ids: List of token IDs
        - response_mask: List of 0/1 where 0 = injected hint token (mask), 1 = model token (train)

    Example:
        >>> text = "reasoning...</think>\\n<request></request>\\n<response>hint</response>\\n<think>\\nmore reasoning"
        >>> tokens, mask = tokenize_with_response_mask(text, tokenizer)
        >>> # "\\n<response>hint</response>\\n<think>\\n" tokens have mask=0, others have mask=1
    """
    # Match the exact RL injection boundary: \n<response>...</response>\n<think>\n
    # The <think>\n suffix is optional for robustness (e.g., "No more hints" edge case)
    pattern = re.compile(
        r'(\n<response>.*?</response>\n(?:<think>\n)?)',
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
        "artifacts/countdown/simple/models/sft/qwen3-4b/model" -> "qwen3-4b"
        "artifacts/countdown/models/sft" -> "sft"
        "/path/to/my-model" -> "my-model"
    """
    # Get last component of path, skipping trailing "model" directory
    parts = model_path.rstrip("/").split("/")
    name = parts[-1]
    if name == "model" and len(parts) > 1:
        name = parts[-2]
    if name == "default":
        name = "qwen2.5-1.5b"
    # Lowercase and clean up
    name = name.lower().replace(" ", "-")
    return name
