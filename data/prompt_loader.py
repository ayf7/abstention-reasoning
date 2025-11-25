from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parent

_TEMPLATE_CACHE: Dict[str, str] = {}


def _get_template(dataset_dir_or_path: str) -> str:
    if dataset_dir_or_path not in _TEMPLATE_CACHE:
        if dataset_dir_or_path.endswith(".txt"):
            # It's a direct path relative to BASE_DIR
            template_path = BASE_DIR / dataset_dir_or_path
        else:
            # It's a dataset directory name
            template_path = BASE_DIR / dataset_dir_or_path / "generation_template.txt"
        _TEMPLATE_CACHE[dataset_dir_or_path] = template_path.read_text(encoding="utf-8")
    return _TEMPLATE_CACHE[dataset_dir_or_path]


def generate_connections_prompt(example: Dict[str, Any]) -> str:
    """
    Build a CoT-style prompt for a single NYT Connections example.
    """
    template = _get_template("connections")
    words = example.get("words") or []
    words_str = ", ".join(words)

    return template.format(
        contest=example.get("contest", ""),
        date=example.get("date", ""),
        words=words_str,
    )


def generate_connections_3x3_prompt(example: Dict[str, Any]) -> str:
    """
    Build a CoT-style prompt for a 3x3 (12 words) NYT Connections example.
    """
    template = _get_template("connections/generation_template_3x3.txt")
    words = example.get("words") or []
    words_str = ", ".join(words)

    return template.format(
        words=words_str,
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

    return template.format(quiz=quiz)


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

    connections_synthetic_path = BASE_DIR / "connections_synthetic" / "train.jsonl"
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
