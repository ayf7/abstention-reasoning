import argparse
import json
import random
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.prompt_loader import generate_knights_and_knaves_prompt

MODEL = "Qwen/Qwen3-8B"
BATCH_SIZE = 16
DEFAULT_SAMPLE_SIZE = 2048
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

def extract_answer(text: str):
    # Look for "Answer: {Name = Role, ...}"
    # We want to capture the content inside braces.
    pattern = r"Answer:\s*\{(.*?)\}"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        # Take the last match
        content = matches[-1]
        # Parse "Name = Role" pairs
        pairs = content.split(",")
        result = {}
        for pair in pairs:
            if "=" in pair:
                name, role = pair.split("=", 1)
                result[name.strip()] = role.strip()
        return result
    return None

def validate_answer(predicted_roles, example):
    if not predicted_roles:
        return False
    
    names = example["names"]
    solution = example["solution"] # List of booleans
    
    # Build ground truth map
    true_roles = {}
    for name, is_knight in zip(names, solution):
        true_roles[name] = "Knight" if is_knight else "Knave"
        
    # Check if predicted roles match true roles
    # We need to make sure all names are present and roles match
    if len(predicted_roles) != len(true_roles):
        return False
        
    for name, role in true_roles.items():
        if name not in predicted_roles:
            return False
        # Case-insensitive comparison for role
        if predicted_roles[name].lower() != role.lower():
            return False
            
    return True

def main(num_samples: int = DEFAULT_SAMPLE_SIZE, seed: int = DEFAULT_SEED):
    tokenizer, model = init_model(MODEL)
    
    knk_path = DATASETS_DIR / "knights_and_knaves" / "train.jsonl"
    output_path = DATASETS_DIR / "knights_and_knaves" / "generations.jsonl"
    
    # Load all examples
    examples = []
    with knk_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if "index" not in data:
                data["index"] = i
            examples.append(data)
            
    total_examples = len(examples)
    rng = random.Random(seed)
    if num_samples is not None and num_samples > 0 and num_samples < total_examples:
        examples = rng.sample(examples, num_samples)
    else:
        num_samples = total_examples

    selected_total = len(examples)
    print(
        f"Processing {selected_total} examples (sampled from {total_examples}) "
        f"with batch size {BATCH_SIZE} and seed {seed}..."
    )
    
    with output_path.open("w", encoding="utf-8") as out_f:
        for i in range(0, selected_total, BATCH_SIZE):
            batch_examples = examples[i : i + BATCH_SIZE]
            prompts = [generate_knights_and_knaves_prompt(ex) for ex in batch_examples]
            
            tokenizer.padding_side = "left" 
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            
            # Check for EOS token in prompt (optional verification)
            eos_id = tokenizer.eos_token_id
            input_ids = inputs["input_ids"]
            for k in range(input_ids.size(0)):
                eos_indices = (input_ids[k] == eos_id).nonzero(as_tuple=True)[0]
                attn_mask = inputs["attention_mask"][k]
                real_eos_indices = [idx.item() for idx in eos_indices if attn_mask[idx].item() == 1]
                if real_eos_indices:
                    print(f"EOS token found in PROMPT at indices {real_eos_indices} for example {batch_examples[k]['index']}")

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            
            generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            for j, full_output in enumerate(generated_texts):
                # Check for EOS token in output
                seq_ids = output_ids[j]
                eos_indices = (seq_ids == eos_id).nonzero(as_tuple=True)[0]
                valid_eos = [idx.item() for idx in eos_indices if idx.item() > 0]
                if valid_eos:
                    print(f"EOS found at indices {valid_eos} for example {batch_examples[j]['index']}")

                ex = batch_examples[j]
                prompt = prompts[j]
                
                if full_output.startswith(prompt):
                    cot = full_output[len(prompt):]
                else:
                    cot = full_output
                
                predicted_roles = extract_answer(cot)
                is_correct = validate_answer(predicted_roles, ex)
                
                record = {
                    "idx": ex["index"],
                    "prompt": prompt,
                    "cot": cot,
                    "correct_answer": is_correct
                }
                
                out_f.write(json.dumps(record) + "\n")
            
            out_f.flush()
            print(f"Processed {min(i + BATCH_SIZE, selected_total)}/{selected_total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Knights and Knaves model outputs.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of questions to sample from the training set (default: {DEFAULT_SAMPLE_SIZE}). "
             "Use 0 or a negative value to process all questions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for sampling the subset of questions (default: {DEFAULT_SEED}).",
    )
    args = parser.parse_args()

    main(num_samples=args.num_samples, seed=args.seed)
