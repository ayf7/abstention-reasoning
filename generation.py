from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prompt_loader import (
    generate_connections_prompt,
    generate_countdown_prompt,
    generate_knights_and_knaves_prompt,
)


BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "data"


def _load_first_jsonl_example(path: Path) -> dict:
    import json

    with path.open("r", encoding="utf-8") as f:
        line = f.readline()
    return json.loads(line)


def init_qwen_model(model_name: str = "Qwen/Qwen3-8B"):
    """
    Initialize the Qwen3 model and tokenizer.

    The default is Qwen/Qwen3-8B; you can change `model_name` if desired.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return tokenizer, model


def generate_cot_responses(tokenizer, model) -> None:
    """
    Generate one CoT response for a single example from each dataset and print it.
    """

    # Build prompts using the same logic as in datasets/prompt_loader.py
    connections_example = _load_first_jsonl_example(
        DATASETS_DIR / "connections" / "train.jsonl"
    )
    countdown_example = _load_first_jsonl_example(
        DATASETS_DIR / "countdown" / "train.jsonl"
    )
    knights_example = _load_first_jsonl_example(
        DATASETS_DIR / "knights_and_knaves" / "train.jsonl"
    )

    prompts = {
        "Connections": generate_connections_prompt(connections_example),
        # "Countdown": generate_countdown_prompt(countdown_example),
        # "Knights and Knaves": generate_knights_and_knaves_prompt(knights_example),
    }

    for name, prompt in prompts.items():
        print(f"=== {name} Prompt ===")
        print(prompt)
        print()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=8192,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode only the newly generated tokens, or you can decode full sequence
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"=== {name} Response ===")
        print(generated_text)
        print()


def main() -> None:
    tokenizer, model = init_qwen_model()
    generate_cot_responses(tokenizer, model)


if __name__ == "__main__":
    main()
