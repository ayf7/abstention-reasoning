"""Shared utility functions."""

import re
from typing import Any


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
