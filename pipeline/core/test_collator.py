"""Test script for segmented tokenization with response masking."""

import json
from transformers import AutoTokenizer

from pipeline.core.utils import tokenize_with_response_mask


def test_masking(dataset_path: str, model_name: str = "Qwen/Qwen2.5-1.5B", example_idx: int = 1):
    """
    Test the segmented tokenization approach for response masking.

    Args:
        dataset_path: Path to the dataset JSON file
        model_name: Model name for tokenizer
        example_idx: Index of example to test (default: 1, which has hints)
    """
    # Load dataset
    with open(dataset_path) as f:
        data = json.load(f)

    # Find an example with hints
    example = data[example_idx]
    print(f"=== Example {example['index']} ===")
    print(f"Hints used: {example.get('num_hints', 0)}")
    print(f"Correct: {example.get('correct', False)}")
    print()

    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print()

    # Format the example as prompt/completion (matching train_sft logic)
    messages = example["prompt"]
    if messages and messages[-1]["role"] == "assistant":
        conversation = messages[:-1]
        assistant_prefix = messages[-1]["content"]
    else:
        conversation = messages
        assistant_prefix = ""

    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    completion = assistant_prefix + example["generation"]

    print(f"=== Prompt (last 200 chars) ===")
    print(repr(prompt[-200:]))
    print()

    print(f"=== Completion (first 500 chars) ===")
    print(repr(completion[:500]))
    print("...")
    print()

    # Tokenize prompt (standard)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    # Tokenize completion with segmented approach
    completion_tokens, response_mask = tokenize_with_response_mask(completion, tokenizer)

    # Combine
    input_ids = prompt_tokens + completion_tokens
    completion_mask = [0] * len(prompt_tokens) + response_mask

    print(f"=== Tokenization ===")
    print(f"Prompt tokens: {len(prompt_tokens)}")
    print(f"Completion tokens: {len(completion_tokens)}")
    print(f"Total tokens: {len(input_ids)}")
    print()

    # Analyze masking
    print(f"=== Masking Analysis ===")
    masked_count = sum(1 for m in completion_mask if m == 0)
    trained_count = sum(1 for m in completion_mask if m == 1)
    print(f"Masked tokens (completion_mask=0): {masked_count}")
    print(f"Trained tokens (completion_mask=1): {trained_count}")
    print()

    # Find contiguous masked regions in the completion part
    print(f"=== Masked Regions (completion only) ===")
    in_masked = False
    start_idx = 0
    masked_regions = []

    # Only look at completion part
    for i, mask in enumerate(response_mask):
        if mask == 0 and not in_masked:
            in_masked = True
            start_idx = i
        elif mask == 1 and in_masked:
            in_masked = False
            masked_regions.append((start_idx, i))

    if in_masked:
        masked_regions.append((start_idx, len(response_mask)))

    print(f"Found {len(masked_regions)} response-masked regions in completion:")
    for i, (start, end) in enumerate(masked_regions):
        region_tokens = completion_tokens[start:end]
        region_text = tokenizer.decode(region_tokens)

        # Truncate for display
        display_text = region_text
        if len(display_text) > 200:
            display_text = display_text[:100] + " ... " + display_text[-100:]

        # Get 3 tokens before and after for context
        ctx_before_start = max(0, start - 3)
        ctx_after_end = min(len(completion_tokens), end + 3)

        tokens_before = completion_tokens[ctx_before_start:start]
        tokens_after = completion_tokens[end:ctx_after_end]

        before_strs = [repr(tokenizer.decode([t])) for t in tokens_before]
        after_strs = [repr(tokenizer.decode([t])) for t in tokens_after]

        print(f"\nRegion {i+1}: tokens [{start}:{end}] ({end-start} tokens)")
        print(f"  3 tokens before: {' '.join(before_strs)}")
        print(f"  3 tokens after:  {' '.join(after_strs)}")
        print(f"  Masked content:  {repr(display_text)}")

    # Verification
    print(f"\n=== Verification ===")

    # Decode trained tokens only
    trained_text = ""
    for token, mask in zip(completion_tokens, response_mask):
        if mask == 1:
            trained_text += tokenizer.decode([token])

    if "<response>" in trained_text:
        print("WARNING: <response> found in trained regions!")
        idx = trained_text.find("<response>")
        print(f"  Context: ...{repr(trained_text[max(0,idx-50):idx+100])}...")
    else:
        print("OK: No <response> tags in trained regions")

    if "</response>" in trained_text:
        print("WARNING: </response> found in trained regions!")
    else:
        print("OK: No </response> tags in trained regions")

    # Check for newlines around response
    if "\n<response>" in trained_text:
        print("WARNING: \\n<response> found in trained regions!")
    else:
        print("OK: No \\n<response> in trained regions")

    if "</response>\n" in trained_text:
        print("WARNING: </response>\\n found in trained regions!")
    else:
        print("OK: No </response>\\n in trained regions")


def find_example_with_hints(data: list, min_hints: int = 2) -> int | None:
    """Find the first example with at least min_hints hints."""
    for i, ex in enumerate(data):
        num_hints = ex.get("num_hints", ex.get("metadata", {}).get("num_hints", 0))
        if num_hints >= min_hints:
            return i
    return None


if __name__ == "__main__":
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/countdown/hint/datasets/sft_qwen3-14b.json"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-1.5B"

    # Find example with hints if not specified
    if len(sys.argv) > 3:
        example_idx = int(sys.argv[3])
    else:
        # Load data to find an example with hints
        with open(dataset_path) as f:
            data = json.load(f)
        example_idx = find_example_with_hints(data, min_hints=2)
        if example_idx is None:
            print("No examples with 2+ hints found, using index 0")
            example_idx = 0
        else:
            print(f"Auto-selected example at index {example_idx} (has 2+ hints)")

    test_masking(dataset_path, model_name, example_idx)
