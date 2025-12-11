import argparse
import json
import random
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.prompt_loader import (
    generate_connections_prompt,
    generate_connections_3x3_prompt,
)

MODEL = "Qwen/Qwen3-8B"
BATCH_SIZE = 8
DEFAULT_SAMPLE_SIZE = None  # Use all examples by default
DEFAULT_SEED = 0
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = BASE_DIR / "data"

def init_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return tokenizer, model

def extract_answer(text: str, variant: str = "standard"):
    if variant == "3x3":
        # Look for 3 groups, optionally enclosed in braces
        pattern = r"Answer:\s*\{?\s*(\[.*?\])\s*,\s*(\[.*?\])\s*,\s*(\[.*?\])\s*\}?"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # Return the last match (to avoid matching the example in the prompt if prompt stripping failed)
            return list(matches[-1])
    else:
        # Look for 4 groups, optionally enclosed in braces
        pattern = r"Answer:\s*\{?\s*(\[.*?\])\s*,\s*(\[.*?\])\s*,\s*(\[.*?\])\s*,\s*(\[.*?\])\s*\}?"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return list(matches[-1])
    return None

def parse_group(group_str: str):
    content = group_str.strip("[]")
    words = [w.strip().upper() for w in content.split(",")]
    return sorted(words)

def validate_answer(predicted_groups_str, true_groups_data):
    print("predicted_groups_str", predicted_groups_str)
    print("true_groups_data", true_groups_data)
    print("----------------")
    if not predicted_groups_str:
        print("False\n----------------")
        return False
    
    try:
        pred_groups = [parse_group(g) for g in predicted_groups_str]
        true_groups = [sorted([w.upper() for w in g["words"]]) for g in true_groups_data]
        
        pred_groups.sort()
        true_groups.sort()
        
        print("pred_groups", pred_groups)
        print("true_groups", true_groups)
        print(pred_groups == true_groups, "\n----------------")
        return pred_groups == true_groups
    except Exception:
        print("False (Exception)\n----------------")
        return False


def clean_cot(text: str, tokenizer) -> tuple[str, int]:
    """
    Trim the generated text to the first Answer block and strip trailing markers.
    Then retokenize to get an accurate token length.
    """
    # Remove anything after an explicit end marker if present.
    if "<|endoftext|>" in text:
        text = text.split("<|endoftext|>", 1)[0]

    # Keep only up to the first "Answer:" block if present.
    answer_match = re.search(r"Answer:\s*\{.*?\}", text, re.DOTALL)
    if answer_match:
        text = text[: answer_match.end()]

    text = text.strip()
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    return text, len(token_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CoT outputs for Connections using the prompt template."
    )
    parser.add_argument(
        "--variant",
        choices=["standard", "3x3"],
        default="standard",
        help="Which prompt variant to use (default: standard).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Optional number of examples to sample (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for sampling (default: 0).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help=f"Model name (default: {MODEL}).",
    )
    return parser.parse_args()

def transform_to_3x3(example):
    # Deep copy to avoid modifying original if needed, but here we just create new dict
    answers = example["answers"]
    # Randomly select 3 indices from 0, 1, 2, 3 (remove 1 group)
    group_indices = sorted(random.sample(range(4), 3))
    
    new_answers = []
    new_words = []
    
    for i in group_indices:
        group = answers[i]
        words = group["words"]
        # Randomly select 3 words from the 4 words in the group (remove 1 word)
        selected_words = sorted(random.sample(words, 3))
        
        new_group = group.copy()
        new_group["words"] = selected_words
        new_answers.append(new_group)
        new_words.extend(selected_words)
    
    # Shuffle the words
    random.shuffle(new_words)
    
    new_example = example.copy()
    new_example["answers"] = new_answers
    new_example["words"] = new_words
    return new_example


def load_processed_indices(path: Path) -> set[int]:
    """
    Read an existing generations file (if any) and collect indices already
    processed so we can resume without regenerating them.
    """
    processed = set()
    if not path.exists():
        return processed

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                if "idx" in record:
                    processed.add(record["idx"])
            except Exception:
                continue
    return processed


def main():
    args = parse_args()
    tokenizer, model = init_model(args.model)
    variant = args.variant
    
    connections_path = DATASETS_DIR / "connections" / "train.jsonl"
    if variant == "3x3":
        output_path = DATASETS_DIR / "connections" / "generations_3x3.jsonl"
    else:
        output_path = DATASETS_DIR / "connections" / "generations.jsonl"
    processed_indices = load_processed_indices(output_path)
    file_mode = "a" if output_path.exists() else "w"
    
    examples = []
    with connections_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if "index" not in data:
                data["index"] = i
            examples.append(data)

    total_examples = len(examples)
    rng = random.Random(args.seed)
    if args.num_samples is not None and 0 < args.num_samples < total_examples:
        examples = rng.sample(examples, args.num_samples)
    selected_total = len(examples)

    print(
        f"Processing {selected_total} examples with batch size {BATCH_SIZE} "
        f"(Variant: {variant}, model={args.model})..."
    )
    
    already_done = len(processed_indices)
    if already_done:
        print(f"Resuming: {already_done} examples already in {output_path}")
        remaining = [ex["index"] for ex in examples if ex["index"] not in processed_indices]
        if remaining:
            print(f"Next unprocessed index: {remaining[0]}")
        else:
            print("All selected examples already processed.")

    with output_path.open(file_mode, encoding="utf-8") as out_f:
        processed_now = already_done
        for i in range(0, selected_total, BATCH_SIZE):
            batch_original = examples[i : i + BATCH_SIZE]
            filtered_examples = [ex for ex in batch_original if ex["index"] not in processed_indices]
            if not filtered_examples:
                continue
            batch_examples = []
            prompts = []
            print(f"Processing indices: {[ex['index'] for ex in filtered_examples]}")
            
            for ex in filtered_examples:
                if variant == "3x3":
                    new_ex = transform_to_3x3(ex)
                    batch_examples.append(new_ex)
                    prompts.append(generate_connections_3x3_prompt(new_ex))
                else:
                    batch_examples.append(ex)
                    prompts.append(generate_connections_prompt(ex))
            
            tokenizer.padding_side = "left" 
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            
            generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            for j, full_output in enumerate(generated_texts):
                ex = batch_examples[j]
                prompt = prompts[j]

                # Slice off the entire input (including padding) to leave only generated tokens.
                input_seq_len = inputs["input_ids"].shape[-1]
                generated_only_ids = output_ids[j][input_seq_len:]
                cot_raw = tokenizer.decode(generated_only_ids, skip_special_tokens=True)
                cot, cot_token_length = clean_cot(cot_raw, tokenizer)
                
                predicted_groups_str = extract_answer(cot, variant=variant)
                is_correct = validate_answer(predicted_groups_str, ex["answers"])
                
                record = {
                    "idx": ex["index"],
                    "prompt": prompt,
                    "cot": cot,
                    "cot_token_length": cot_token_length,
                    "correct_answer": is_correct
                }
                
                out_f.write(json.dumps(record) + "\n")
                processed_indices.add(ex["index"])
            
            out_f.flush()
            processed_now += len(filtered_examples)
            print(f"Processed {processed_now}/{selected_total}")


if __name__ == "__main__":
    main()
