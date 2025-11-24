from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parent

_TEMPLATE_CACHE: Dict[str, str] = {}


def _get_template(dataset_dir: str) -> str:
    if dataset_dir not in _TEMPLATE_CACHE:
        template_path = BASE_DIR / dataset_dir / "generation_template.txt"
        _TEMPLATE_CACHE[dataset_dir] = template_path.read_text(encoding="utf-8")
    return _TEMPLATE_CACHE[dataset_dir]


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


def _load_first_jsonl_example(path: Path) -> Dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as f:
        line = f.readline()
    return json.loads(line)


def main() -> None:
    """
    Load the first training example from each dataset and print the
    corresponding CoT-style prompt. This is intended as a quick sanity
    check that the templates and generators are wired up correctly.
    """

    # Connections
    connections_path = BASE_DIR / "connections" / "train.jsonl"
    connections_example = _load_first_jsonl_example(connections_path)
    connections_prompt = generate_connections_prompt(connections_example)
    print("=== Connections Prompt ===")
    print(connections_prompt)
    print()

    # Countdown
    countdown_path = BASE_DIR / "countdown" / "train.jsonl"
    countdown_example = _load_first_jsonl_example(countdown_path)
    countdown_prompt = generate_countdown_prompt(countdown_example)
    print("=== Countdown Prompt ===")
    print(countdown_prompt)
    print()

    # Knights and Knaves
    knk_path = BASE_DIR / "knights_and_knaves" / "train.jsonl"
    knk_example = _load_first_jsonl_example(knk_path)
    knk_prompt = generate_knights_and_knaves_prompt(knk_example)
    print("=== Knights and Knaves Prompt ===")
    print(knk_prompt)


if __name__ == "__main__":
    main()
