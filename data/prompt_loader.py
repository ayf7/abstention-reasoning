from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parent

_TEMPLATE_CACHE: Dict[str, str] = {}


def _get_template(dataset_dir_or_path: str | Path) -> str:
    key = str(dataset_dir_or_path)
    if key not in _TEMPLATE_CACHE:
        if key.endswith(".txt"):
            # It's a direct path relative to BASE_DIR
            template_path = BASE_DIR / key
        else:
            # It's a dataset directory name
            template_path = BASE_DIR / key / "generation_template.txt"
        _TEMPLATE_CACHE[key] = template_path.read_text(encoding="utf-8")
    return _TEMPLATE_CACHE[key]


def _compute_connection_counts(
    example: Dict[str, Any],
    group_count: int | None = None,
    words_per_group: int | None = None,
) -> tuple[int, int, int]:
    answers = example.get("answers") or []
    words = example.get("words") or []

    derived_group_count = group_count or example.get("group_count")
    if derived_group_count is None:
        derived_group_count = len(answers) if answers else 0

    derived_words_per_group = words_per_group or example.get("words_per_group")
    if derived_words_per_group is None:
        if answers and isinstance(answers, list) and answers[0].get("words"):
            derived_words_per_group = len(answers[0]["words"])
        elif derived_group_count:
            derived_words_per_group = len(words) // derived_group_count if words else 0
        else:
            derived_words_per_group = len(words)

    total_words = (
        len(words)
        if words
        else example.get("total_words")
        or derived_group_count * derived_words_per_group
    )

    return (
        derived_group_count or 0,
        derived_words_per_group or 0,
        total_words or 0,
    )


def _build_example_answer(group_count: int, words_per_group: int) -> str:
    group = "[" + ", ".join("<word>" for _ in range(words_per_group)) + "]"
    groups_str = ", ".join(group for _ in range(group_count))
    return "{" + groups_str + "}"


def generate_connections_prompt(
    example: Dict[str, Any],
    *,
    template_path: str | Path | None = None,
    group_count: int | None = None,
    words_per_group: int | None = None,
) -> str:
    """
    Build a CoT-style prompt for a single NYT Connections example.
    """
    template_key = template_path or "connections"
    template = _get_template(template_key)
    words = example.get("words") or []
    words_str = ", ".join(words)
    group_count, words_per_group, total_words = _compute_connection_counts(
        example,
        group_count=group_count,
        words_per_group=words_per_group,
    )
    example_answer = _build_example_answer(group_count, words_per_group)

    return template.format(
        contest=example.get("contest", ""),
        date=example.get("date", ""),
        words=words_str,
        group_count=group_count,
        words_per_group=words_per_group,
        total_words=total_words,
        example_answer=example_answer,
    )


def generate_countdown_prompt(example: Dict[str, Any]) -> str:
    """
    Build a CoT-style prompt for a single Countdown numbers example.
    """
    template = _get_template("countdown")
    nums = example.get("nums") or []
    nums_str = ", ".join(str(n) for n in nums)

    return template.format(
        target=example.get("target", ""),
        nums=nums_str,
    )


def generate_knights_and_knaves_prompt(example: Dict[str, Any]) -> str:
    """
    Build a CoT-style prompt for a single Knights and Knaves logic puzzle.
    """
    template = _get_template("knights_and_knaves")
    quiz = example.get("quiz", "")
    names = example.get("names") or []
    length = len(names)

    return template.format(quiz=quiz, length=length)


def load_jsonl_example(path: Path, index: int = 0) -> Dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                data = json.loads(line)
                data["index"] = index
                return data
    
    raise IndexError(f"Index {index} out of range for file {path}")


def main() -> None:
    """
    Load the first training example from each dataset and print the
    corresponding CoT-style prompt. This is intended as a quick sanity
    check that the templates and generators are wired up correctly.
    """

    # Connections
    connections_path = BASE_DIR / "connections" / "train.jsonl"
    connections_example = load_jsonl_example(connections_path, index=0)
    connections_prompt = generate_connections_prompt(connections_example)
    print("=== Connections Prompt ===")
    print(connections_prompt)
    print()

    connections_synthetic_path = BASE_DIR / "connections_synthetic" / "train_3x2.jsonl"
    connections_synthetic_example = load_jsonl_example(connections_synthetic_path, index=0)
    connections_synthetic_prompt = generate_connections_prompt(connections_synthetic_example)
    print("=== Connections (synthetic) Prompt ===")
    print(connections_synthetic_prompt)
    print()

    # Countdown
    # countdown_path = BASE_DIR / "countdown" / "train.jsonl"
    # countdown_example = load_jsonl_example(countdown_path, index=0)
    # countdown_prompt = generate_countdown_prompt(countdown_example)
    # print("=== Countdown Prompt ===")
    # print(countdown_prompt)
    # print()

    # Knights and Knaves
    knk_path = BASE_DIR / "knights_and_knaves" / "train.jsonl"
    knk_example = load_jsonl_example(knk_path, index=0)
    knk_prompt = generate_knights_and_knaves_prompt(knk_example)
    print("=== Knights and Knaves Prompt ===")
    print(knk_prompt)


if __name__ == "__main__":
    main()
