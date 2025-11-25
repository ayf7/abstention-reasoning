import json
import re
import torch
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.prompt_loader import load_jsonl_example, generate_connections_prompt, generate_connections_3x3_prompt

MODEL = "Qwen/Qwen3-8B"
BATCH_SIZE = 8
VARIANT = "3x3" # "standard" or "3x3"
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

def main():
    tokenizer, model = init_model(MODEL)
    
    connections_path = DATASETS_DIR / "connections" / "train.jsonl"
    if VARIANT == "3x3":
        output_path = DATASETS_DIR / "connections" / "generations_3x3.jsonl"
    else:
        output_path = DATASETS_DIR / "connections" / "generations.jsonl"
    
    examples = []
    with connections_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if "index" not in data:
                data["index"] = i
            examples.append(data)
            
    total_examples = len(examples)
    print(f"Processing {total_examples} examples with batch size {BATCH_SIZE} (Variant: {VARIANT})...")
    
    with output_path.open("w", encoding="utf-8") as out_f:
        for i in range(0, total_examples, BATCH_SIZE):
            batch_original = examples[i : i + BATCH_SIZE]
            batch_examples = []
            prompts = []
            
            for ex in batch_original:
                if VARIANT == "3x3":
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
                    max_new_tokens=8192,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            
            generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            for j, full_output in enumerate(generated_texts):
                # Check for EOS token
                seq_ids = output_ids[j]
                eos_id = tokenizer.eos_token_id
                # Find indices where eos_id appears
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
                
                predicted_groups_str = extract_answer(cot, variant=VARIANT)
                is_correct = validate_answer(predicted_groups_str, ex["answers"])
                
                record = {
                    "idx": ex["index"],
                    "prompt": prompt,
                    "cot": cot,
                    "correct_answer": is_correct
                }
                
                out_f.write(json.dumps(record) + "\n")
            
            out_f.flush()
            print(f"Processed {min(i + BATCH_SIZE, total_examples)}/{total_examples}")

if __name__ == "__main__":
    main()
