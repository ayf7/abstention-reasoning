"""Shared utility functions."""

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ForceAnswer:
    """Answer forcing: separate token budgets for reasoning and the answer.

    A generation that spends think_tokens without answering gets </think>
    closed for it and answer_tokens more to answer in. If the forced answer
    also runs out, the sample is incomplete. A model that closes </think> on
    its own answers inside think_tokens, as RL rollouts do.
    """
    think_tokens: int
    answer_tokens: int

    @classmethod
    def parse(cls, spec: str) -> "ForceAnswer":
        """From JSON, e.g. '{"think_tokens": 2048, "answer_tokens": 512}'."""
        try:
            data = json.loads(spec)
        except json.JSONDecodeError as e:
            raise ValueError(f"not valid JSON ({e}): {spec}") from None
        keys = {"think_tokens", "answer_tokens"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError(f"needs exactly the keys {sorted(keys)}, got: {spec}")
        for key in keys:
            value = data[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer, got {value!r}")
        return cls(**data)


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


# What the generator inserts when it forces an answer out of a truncated
# generation (_answer_scaffold's mid-think case), plus the newline that opens a
# multi-line answer body after canonicalization. A one-line answer such as a
# verifier's 0/1 label has no such newline.
FORCED_ANSWER_SCAFFOLD = "\n</think>\n\n<answer>"


def tokenize_with_forced_scaffold_mask(
    text: str,
    tokenizer,
) -> tuple[list[int], list[int]]:
    """Tokenize a forced-answer completion with the inserted scaffold masked.

    The model never wrote the scaffold, so it gets mask 0 and is excluded from
    the loss. It is tokenized as its own segment, matching how RL splices it
    in at rollout time. The last occurrence is the inserted one: the scaffold
    closes the think block, so nothing after it is reasoning.
    """
    head, scaffold, tail = text.rpartition(FORCED_ANSWER_SCAFFOLD)
    if not scaffold:
        raise ValueError("forced_answer record has no forced-answer scaffold")
    if tail.startswith("\n"):
        scaffold, tail = scaffold + "\n", tail[1:]
    tokens, mask = [], []
    for segment, value in ((head, 1), (scaffold, 0), (tail, 1)):
        ids = tokenizer.encode(segment, add_special_tokens=False)
        tokens.extend(ids)
        mask.extend([value] * len(ids))
    return tokens, mask


def extract_answer(text: str) -> str | None:
    """Extract content between <answer> and </answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


_AFTER_HINT_END = re.compile(r"<request>|</think>|<answer>")


def min_words_after_hints(text: str) -> int | None:
    """Fewest words of reasoning written after any hint, or None if no hint was served.

    A segment runs from a </response> to the next request, the end of thinking,
    or the answer. A teacher that chains requests or answers straight off a hint
    scores 0.
    """
    segments = text.split("</response>")[1:]
    if not segments:
        return None
    return min(len(_AFTER_HINT_END.split(s.replace("<think>", ""), 1)[0].split()) for s in segments)


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
