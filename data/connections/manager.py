"""
Standardized manager for NYT Connections dataset.

Manages the full pipeline from groups.jsonl + synthetic_groups.jsonl ->
raw_dataset -> train/test splits -> CoT generation -> SFT training ->
RL data creation.

Supports multiple puzzle variants: 2x3, 3x2, 3x3.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Don't import torch or torch-dependent libraries here to avoid CUDA initialization issues with VLLM multiprocessing
# import torch
import pandas as pd
from datasets import load_dataset
# from trl import SFTTrainer, SFTConfig  # TRL imports torch - lazy import in run_sft instead
from vllm import LLM, SamplingParams

from data.dataset_manager import DatasetManager
from data.prompt_loader import generate_connections_prompt


class ConnectionsManager(DatasetManager):
    """
    Manages the Connections dataset following the standardized framework.

    Directory structure:
        data/connections/
            groups.jsonl             # Original groups from CSV (expected to exist)
            synthetic_groups.jsonl   # Synthetic groups from OpenAI (expected to exist)
            generation_template.txt  # Prompt template (expected to exist)
            artifacts/
                raw_dataset.jsonl
                train.jsonl
                test.jsonl
                rl_train.parquet
                rl_val.parquet
    """

    def __init__(
        self,
        path: Path = None,
        prompt_template: Path = None,
        groups_path: Path = None,
        synthetic_groups_path: Path = None,
        test_split_ratio: float = 0.2,
        seed: int = 42,
        count_2x3: int = 500,
        count_3x2: int = 500,
        count_3x3: int = 500,
    ):
        if path is None:
            path = Path(__file__).resolve().parent
        if prompt_template is None:
            prompt_template = path / "generation_template.txt"
        if groups_path is None:
            groups_path = path / "groups.jsonl"
        if synthetic_groups_path is None:
            synthetic_groups_path = path / "synthetic_groups.jsonl"

        super().__init__(path, prompt_template)

        self.groups_path = groups_path
        self.synthetic_groups_path = synthetic_groups_path
        self.test_split_ratio = test_split_ratio
        self.seed = seed
        self.rng = random.Random(seed)
        self.count_2x3 = count_2x3
        self.count_3x2 = count_3x2
        self.count_3x3 = count_3x3

    # ========================================================================
    # Dataset Creation
    # ========================================================================

    def create_dataset(self):
        """
        Creates raw_dataset.jsonl from groups.jsonl and synthetic_groups.jsonl.

        Each record in raw_dataset.jsonl:
        {
            "index": <int>,
            "question": {
                "words": <list of shuffled words>,
                "answers": [{"answerDescription": <str>, "words": <list>}, ...]
            },
            "answer": <str>,  # Canonical answer format
            "metadata": {
                "variant": <str>,  # e.g., "2x3", "3x2", "3x3"
                "total_words": <int>,
                "group_count": <int>,
                "words_per_group": <int>,
                "indexes_used": <list of group indices used>
            }
        }
        """
        if not self.groups_path.exists():
            raise FileNotFoundError(f"Groups file not found: {self.groups_path}")
        if not self.synthetic_groups_path.exists():
            raise FileNotFoundError(f"Synthetic groups file not found: {self.synthetic_groups_path}")

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.artifact_dir / "raw_dataset.jsonl"

        print(f"Creating dataset from {self.groups_path} and {self.synthetic_groups_path}...")

        # Load groups from both sources
        orig_pool = self._load_groups(self.groups_path, source="orig")
        synth_pool = self._load_groups(self.synthetic_groups_path, source="synth")

        if not orig_pool:
            raise ValueError("No original groups loaded.")
        if not synth_pool:
            raise ValueError("No synthetic groups loaded.")

        print(f"Loaded {len(orig_pool)} original groups and {len(synth_pool)} synthetic groups")

        # Shuffle pools
        self.rng.shuffle(orig_pool)
        self.rng.shuffle(synth_pool)

        # Build puzzles for each variant
        records = []

        print(f"Generating {self.count_2x3} puzzles with 2x3 variant...")
        records.extend(self._generate_variant(2, 3, self.count_2x3, orig_pool, synth_pool))

        print(f"Generating {self.count_3x2} puzzles with 3x2 variant...")
        records.extend(self._generate_variant(3, 2, self.count_3x2, orig_pool, synth_pool))

        print(f"Generating {self.count_3x3} puzzles with 3x3 variant...")
        records.extend(self._generate_variant(3, 3, self.count_3x3, orig_pool, synth_pool))

        # Assign sequential indices
        for idx, rec in enumerate(records):
            rec["index"] = idx

        # Write to JSONL
        with output_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"Created {len(records)} puzzles in {output_path}")
        return records

    def _load_groups(self, path: Path, source: str) -> List[dict]:
        """Load groups from a JSONL file."""
        groups = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                desc = rec.get("description")
                words = rec.get("words")
                rec_idx = rec.get("index")
                if not isinstance(desc, str) or not isinstance(words, list):
                    continue
                cleaned_words = [str(w).strip().upper() for w in words if str(w).strip()]
                if len(cleaned_words) < 2:
                    continue
                groups.append({
                    "description": desc.strip(),
                    "words": cleaned_words,
                    "source": source,
                    "idx": rec_idx if isinstance(rec_idx, int) else None,
                })
        return groups

    def _generate_variant(
        self,
        group_count: int,
        words_per_group: int,
        count: int,
        orig_pool: List[dict],
        synth_pool: List[dict]
    ) -> List[dict]:
        """Generate puzzles for a specific variant."""
        records = []
        attempts = 0
        while len(records) < count and attempts < count * 100:
            attempts += 1
            result = self._build_puzzle(group_count, words_per_group, orig_pool, synth_pool)
            if result is None:
                continue
            orig_used, synth_used, puzzle_record = result
            # Remove used groups
            self._pop_indices(orig_pool, orig_used)
            self._pop_indices(synth_pool, synth_used)
            records.append(puzzle_record)

        if len(records) < count:
            print(f"Warning: generated {len(records)}/{count} for variant {group_count}x{words_per_group}")

        return records

    def _build_puzzle(
        self,
        group_count: int,
        words_per_group: int,
        orig_pool: List[dict],
        synth_pool: List[dict]
    ) -> Optional[Tuple[List[int], List[int], dict]]:
        """Build a single puzzle with the specified variant."""
        if not orig_pool or not synth_pool:
            return None

        used_words = set()
        answers = []
        orig_used = []
        synth_used = []
        indexes_used = []

        # Sample one from each source first
        first = self._sample_group(orig_pool, words_per_group, used_words)
        second = self._sample_group(synth_pool, words_per_group, used_words) if first else None

        if not first or not second:
            return None

        for idx, grp, sampled in (first, second):
            used_words.update(sampled)
            answers.append({"answerDescription": grp["description"], "words": sampled})
            if grp["source"] == "orig":
                orig_used.append(idx)
            else:
                synth_used.append(idx)
            if isinstance(grp.get("idx"), int):
                indexes_used.append(grp["idx"])

        # Add additional groups until reaching group_count
        combined = list(enumerate(orig_pool)) + [(len(orig_pool) + i, g) for i, g in enumerate(synth_pool)]
        self.rng.shuffle(combined)

        while len(answers) < group_count and combined:
            global_idx, grp = combined.pop()
            if len(grp["words"]) < words_per_group:
                continue
            sampled = self.rng.sample(grp["words"], words_per_group)
            if any(w in used_words for w in sampled):
                continue
            used_words.update(sampled)
            answers.append({"answerDescription": grp["description"], "words": sampled})
            if global_idx < len(orig_pool):
                orig_used.append(global_idx)
            else:
                synth_used.append(global_idx - len(orig_pool))
            if isinstance(grp.get("idx"), int):
                indexes_used.append(grp["idx"])

        # Require at least one orig and one synth group
        if not orig_used or not synth_used or len(answers) != group_count:
            return None

        # Shuffle answers and words
        self.rng.shuffle(answers)
        words_flat = []
        for ans in answers:
            words_flat.extend(ans["words"])
        self.rng.shuffle(words_flat)

        # Build standardized record
        variant = f"{group_count}x{words_per_group}"
        canonical_answer = self._format_canonical_answer(answers)

        record = {
            "index": 0,  # Will be set later
            "question": {
                "words": words_flat,
                "answers": answers,
            },
            "answer": canonical_answer,
            "metadata": {
                "variant": variant,
                "total_words": len(words_flat),
                "group_count": group_count,
                "words_per_group": words_per_group,
                "indexes_used": sorted(indexes_used),
            }
        }

        return orig_used, synth_used, record

    def _sample_group(
        self,
        pool: List[dict],
        size: int,
        used_words: set
    ) -> Optional[Tuple[int, dict, List[str]]]:
        """Sample a group from the pool with the required size."""
        if not pool:
            return None
        for _ in range(200):
            idx = self.rng.randrange(len(pool))
            grp = pool[idx]
            if len(grp["words"]) < size:
                continue
            sampled = self.rng.sample(grp["words"], size)
            if any(w in used_words for w in sampled):
                continue
            return idx, grp, sampled
        return None

    @staticmethod
    def _pop_indices(lst: List[dict], indices: List[int]) -> None:
        """Remove elements at specified indices from a list."""
        for idx in sorted(indices, reverse=True):
            lst.pop(idx)

    def _format_canonical_answer(self, answers: List[dict]) -> str:
        """Format the canonical answer string."""
        groups = []
        for ans in answers:
            words_upper = [w.upper() for w in ans["words"]]
            group_str = "[" + ", ".join(words_upper) + "]"
            groups.append(group_str)
        return "{" + ", ".join(groups) + "}"

    # ========================================================================
    # Train/Test Split
    # ========================================================================

    def create_split(self):
        """
        Creates train.jsonl and test.jsonl from raw_dataset.jsonl.

        Each record in train/test.jsonl:
        {
            "prompt": <str or list of messages>,
            "cot": "",  # Initially empty
            "cot_metadata": {},  # Initially empty
            "index": <int>
        }
        """
        raw_path = self.artifact_dir / "raw_dataset.jsonl"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"{raw_path} not found. Run create_dataset() first."
            )

        print(f"Creating train/test split from {raw_path}...")

        # Load raw dataset
        records = []
        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        # Shuffle and split
        self.rng.shuffle(records)
        test_size = int(len(records) * self.test_split_ratio)
        test_records = records[:test_size]
        train_records = records[test_size:]

        # Convert to train/test format
        train_split = [self._to_split_format(rec) for rec in train_records]
        test_split = [self._to_split_format(rec) for rec in test_records]

        # Write splits
        train_path = self.artifact_dir / "train.jsonl"
        test_path = self.artifact_dir / "test.jsonl"

        self._write_jsonl(train_path, train_split)
        self._write_jsonl(test_path, test_split)

        print(f"Created {len(train_split)} train examples in {train_path}")
        print(f"Created {len(test_split)} test examples in {test_path}")

        return train_split, test_split

    def _to_split_format(self, record: dict) -> dict:
        """Convert raw record to train/test format with prompt."""
        # Build example for prompt generation
        example = {
            "words": record["question"]["words"],
            "answers": record["question"]["answers"],
            "contest": record["metadata"].get("contest", ""),
            "group_count": record["metadata"]["group_count"],
            "words_per_group": record["metadata"]["words_per_group"],
            "total_words": record["metadata"]["total_words"],
        }

        # Generate prompt
        prompt = generate_connections_prompt(
            example,
            template_path=self.prompt_template
        )

        return {
            "prompt": prompt,
            "cot": "",
            "cot_metadata": {},
            "index": record["index"],
        }

    @staticmethod
    def _write_jsonl(path: Path, records: List[dict]) -> None:
        """Write records to JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ========================================================================
    # CoT Generation
    # ========================================================================

    def create_generations(
        self,
        in_place: bool = True,
        output_file: Path | None = None,
        model_name: str = "Qwen/Qwen2.5-3B",
        batch_size: int = 8,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_samples: int | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Creates CoT generations for train.jsonl.

        Updates train.jsonl in-place with generated CoTs and metadata.
        Supports resumption if interrupted.
        """
        train_path = self.artifact_dir / "train.jsonl"
        if not train_path.exists():
            raise FileNotFoundError(
                f"{train_path} not found. Run create_split() first."
            )

        print(f"Generating CoTs for {train_path}...")
        print(f"Model: {model_name}, Batch size: {batch_size}, Max tokens: {max_new_tokens}")
        print(f"Tensor parallel size: {tensor_parallel_size} GPU(s)")

        # Load model with VLLM
        model = self._init_model(model_name, tensor_parallel_size, gpu_memory_utilization)

        # Load examples
        examples = []
        with train_path.open("r", encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line))

        # Sample if requested
        if num_samples and 0 < num_samples < len(examples):
            examples = self.rng.sample(examples, num_samples)

        # Find already processed examples
        processed_indices = set()
        if in_place:
            processed_indices = self._find_processed_indices(examples)

        # Generate in batches
        print(f"Processing {len(examples)} examples...")
        if processed_indices:
            print(f"Resuming: {len(processed_indices)} already processed")

        # Create index map for fast lookups
        examples_by_index = {ex["index"]: ex for ex in examples}

        total_processed = len(processed_indices)
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]

            # Filter already processed
            batch = [ex for ex in batch if ex["index"] not in processed_indices]
            if not batch:
                continue

            print(f"Processing batch {i//batch_size + 1}: indices {[ex['index'] for ex in batch]}")

            # Generate
            prompts = [ex["prompt"] for ex in batch]
            generated_texts = self._generate_batch(
                model, prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )

            # Process generations and update in-place
            for j, (ex, gen_text) in enumerate(zip(batch, generated_texts)):
                cot, cot_length = self._clean_cot(gen_text, model)

                # Check correctness (load raw dataset to get answer)
                is_correct = self._check_correctness(ex["index"], cot)

                # Update the example in the dict
                examples_by_index[ex["index"]]["cot"] = cot
                examples_by_index[ex["index"]]["cot_metadata"] = {
                    "correct_answer": is_correct,
                    "cot_token_length": cot_length,
                }
                processed_indices.add(ex["index"])

            # Write immediately after each batch
            if in_place:
                # Reconstruct the list in original order
                all_examples = [examples_by_index[ex["index"]] for ex in examples]
                self._write_jsonl(train_path, all_examples)
                batch_count = len(batch)
                total_processed += batch_count
                print(f"Saved batch to {train_path} ({batch_count} new, {total_processed} total processed)")

        # Final write if output_file specified
        if output_file and not in_place:
            updated_examples = [ex for ex in examples_by_index.values() if ex.get("cot")]
            self._write_jsonl(output_file, updated_examples)
            print(f"Wrote {len(updated_examples)} generations to {output_file}")

        # Return all examples with generations
        return [ex for ex in examples_by_index.values() if ex.get("cot")]

    def _init_model(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9
    ) -> LLM:
        """Initialize VLLM model."""
        print(f"Loading model with VLLM: {model_name}...")
        model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        return model

    def _generate_batch(
        self,
        model: LLM,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[str]:
        """Generate text for a batch of prompts using VLLM."""
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )

        outputs = model.generate(prompts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]

        return generated_texts

    def _clean_cot(self, text: str, model: LLM) -> Tuple[str, int]:
        """Clean generated text and compute token length."""
        # Remove end markers
        if "<|endoftext|>" in text:
            text = text.split("<|endoftext|>", 1)[0]

        # Keep only up to the answer block
        answer_match = re.search(r"<answer>.*?</answer>", text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            text = text[:answer_match.end()]

        text = text.strip()

        # Use VLLM's tokenizer to compute token length
        tokenizer = model.get_tokenizer()
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return text, len(token_ids)

    def _check_correctness(self, index: int, cot: str) -> bool:
        """Check if the generated CoT has the correct answer."""
        # Load raw dataset to get ground truth
        raw_path = self.artifact_dir / "raw_dataset.jsonl"
        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record["index"] == index:
                    ground_truth = record["answer"]
                    predicted = self._extract_answer(cot)
                    return self._compare_answers(predicted, ground_truth, record["question"]["answers"])
        return False

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from generated text."""
        # Look for <answer> tags
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: look for Answer: pattern
        match = re.search(r"Answer:\s*(\{.*?\})", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return None

    def _compare_answers(
        self,
        predicted: Optional[str],
        ground_truth: str,
        answers: List[dict]
    ) -> bool:
        """Compare predicted answer with ground truth."""
        if not predicted:
            return False

        # Normalize both
        pred_norm = self._normalize_answer(predicted)
        gt_norm = self._normalize_answer(ground_truth)

        if pred_norm == gt_norm:
            return True

        # Try parsing as groups
        try:
            pred_groups = self._parse_answer_groups(predicted)
            true_groups = [sorted([w.upper() for w in g["words"]]) for g in answers]

            pred_groups.sort()
            true_groups.sort()

            return pred_groups == true_groups
        except Exception:
            return False

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """Normalize answer for comparison."""
        # Remove whitespace, lowercase
        return re.sub(r"\s+", "", answer.lower())

    @staticmethod
    def _parse_answer_groups(answer: str) -> List[List[str]]:
        """Parse answer into groups of words."""
        # Extract all groups [word1, word2, ...]
        groups = []
        pattern = r"\[(.*?)\]"
        matches = re.findall(pattern, answer, re.DOTALL)
        for match in matches:
            words = [w.strip().upper() for w in match.split(",")]
            groups.append(sorted(words))
        return groups

    def _find_processed_indices(self, examples: List[dict]) -> set:
        """Find indices that already have CoT generations."""
        processed = set()
        for ex in examples:
            if ex.get("cot") and ex.get("cot_metadata"):
                processed.add(ex["index"])
        return processed

    def _merge_examples(
        self,
        original: List[dict],
        updated: List[dict]
    ) -> List[dict]:
        """Merge updated examples back into original list."""
        # Create index map
        updated_map = {ex["index"]: ex for ex in updated}

        # Merge
        result = []
        for ex in original:
            if ex["index"] in updated_map:
                result.append(updated_map[ex["index"]])
            else:
                result.append(ex)

        return result

    # ========================================================================
    # SFT Training
    # ========================================================================

    def run_sft(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        output_dir: str = "checkpoints/connections_sft",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        eval_steps: int = 50,
        save_steps: int = 100,
    ):
        """
        Runs SFT on correct CoT examples from train.jsonl.
        """
        train_path = self.artifact_dir / "train.jsonl"
        if not train_path.exists():
            raise FileNotFoundError(
                f"{train_path} not found. Run create_split() first."
            )

        print(f"Running SFT on {train_path}...")

        # Load dataset and filter for correct answers
        dataset = load_dataset("json", data_files=str(train_path))["train"]

        def has_correct_cot(example):
            cot_meta = example.get("cot_metadata", {})
            return (
                cot_meta.get("correct_answer") is True
                and isinstance(cot_meta.get("cot_token_length"), int)
                and example.get("cot", "").strip()
            )

        filtered = dataset.filter(has_correct_cot)
        print(f"Filtered to {len(filtered)} correct examples")

        if len(filtered) == 0:
            raise ValueError("No correct examples found. Run create_generations() first.")

        # Preprocess for SFT
        debug_printed = [False]  # Use list to allow modification in nested function

        def preprocess(example):
            # Split the prompt to separate assistant prefix
            orig_prompt = example["prompt"]
            orig_cot = example["cot"]

            prompt = orig_prompt
            completion = orig_cot

            # Debug print for first example
            if not debug_printed[0]:
                print("\n" + "="*80)
                print("DEBUG: First example before/after preprocessing")
                print("="*80)
                print(f"ORIGINAL PROMPT (last 200 chars):\n{repr(orig_prompt[-200:])}\n")
                print(f"ORIGINAL COT (first 100 chars):\n{repr(orig_cot[:100])}\n")

            # The prompt ends with "Assistant: <think> Let me solve this step by step."
            # We want to keep everything up to "Assistant: " in prompt
            # and prepend the assistant prefix to the completion with a space for proper grammar
            if "\n\nAssistant: " in prompt:
                parts = prompt.rsplit("\n\nAssistant: ", 1)
                prompt = parts[0] + "\n\nAssistant: "
                assistant_prefix = parts[1]  # "<think> Let me solve this step by step."
                completion = assistant_prefix + completion

            # Continue debug print
            if not debug_printed[0]:
                print(f"NEW PROMPT (last 100 chars):\n{repr(prompt[-100:])}\n")
                print(f"NEW COMPLETION (first 150 chars):\n{repr(completion[:150])}\n")
                print(f"CONCATENATED (relevant portion):\n{repr((prompt[-50:] + completion[:100]))}\n")
                print("="*80 + "\n")
                debug_printed[0] = True

            return {
                "prompt": prompt,
                "completion": completion
            }

        processed = filtered.map(
            preprocess,
            remove_columns=["cot", "cot_metadata"]
        )

        # Print one example for debugging
        if len(processed) > 0:
            print("\n" + "="*80)
            print("Sample training example (first example):")
            print("="*80)
            example = processed[0]
            print(f"PROMPT:\n{example['prompt']}")
            print(f"\nCOMPLETION:\n{example['completion'][:500]}..." if len(example['completion']) > 500 else f"\nCOMPLETION:\n{example['completion']}")
            print("="*80 + "\n")

        # Split for validation
        split_dataset = processed.train_test_split(test_size=0.1, seed=self.seed)

        # Setup training
        # Lazy import torch and TRL to avoid CUDA initialization issues with VLLM
        import torch
        from trl import SFTTrainer, SFTConfig

        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            save_total_limit=3,
            logging_steps=10,
            report_to="wandb",
            bf16=torch.cuda.is_available(),
        )

        trainer = SFTTrainer(
            model=model_name,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
        )

        print("Starting SFT training...")
        trainer.train()

        print(f"Training complete. Model saved to {output_dir}")
        return trainer

    # ========================================================================
    # RL Data Creation
    # ========================================================================

    def create_rl_data(
        self,
        train_split_ratio: float = 0.8,
    ):
        """
        Creates rl_train.parquet and rl_val.parquet from train.jsonl.

        Each record in RL parquet:
        {
            "prompt": <list of messages>,
            "reward_model": {
                "ground_truth": {
                    "solution_text": <str>,
                    "answers": [<list of acceptable formats>],
                }
            },
            "metadata": <dict>,
            "index": <int>
        }
        """
        train_path = self.artifact_dir / "train.jsonl"
        if not train_path.exists():
            raise FileNotFoundError(
                f"{train_path} not found. Run create_split() first."
            )

        raw_path = self.artifact_dir / "raw_dataset.jsonl"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"{raw_path} not found. Run create_dataset() first."
            )

        print(f"Creating RL data from {train_path}...")

        # Load raw dataset for ground truth
        raw_data = {}
        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                raw_data[record["index"]] = record

        # Load train examples
        examples = []
        with train_path.open("r", encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line))

        # Convert to RL format
        rl_records = []
        for ex in examples:
            idx = ex["index"]
            if idx not in raw_data:
                continue

            raw_record = raw_data[idx]

            # Convert prompt to chat messages format
            # The prompt is formatted as: "System: ...\n\nUser: ...\n\nAssistant: ..."
            prompt_text = ex["prompt"]

            # Parse the formatted prompt
            try:
                parts = prompt_text.split("\n\n")
                system_part = parts[0].replace("System: ", "")
                user_part = parts[1].replace("User: ", "")
                assistant_part = parts[2].replace("Assistant: ", "")

                messages = [
                    {"role": "system", "content": system_part},
                    {"role": "user", "content": user_part},
                    {"role": "assistant", "content": assistant_part}
                ]
            except:
                # Fallback for old format
                messages = [
                    {"role": "user", "content": prompt_text}
                ]

            # Build reward model data
            reward_data = {
                "ground_truth": {
                    "solution_text": raw_record["answer"],
                    "answers": raw_record["question"]["answers"],
                }
            }

            rl_record = {
                "prompt": messages,
                "reward_model": reward_data,
                "metadata": raw_record["metadata"],
                "index": idx,
            }

            rl_records.append(rl_record)

        # Shuffle and split
        self.rng.shuffle(rl_records)
        train_size = int(len(rl_records) * train_split_ratio)
        rl_train = rl_records[:train_size]
        rl_val = rl_records[train_size:]

        # Convert to DataFrame and save as parquet
        # Important: Convert nested structures to proper format for parquet
        import pyarrow as pa
        import pyarrow.parquet as pq

        def prepare_records_for_parquet(records):
            """Convert records to format suitable for parquet with nested structures."""
            prepared = []
            for rec in records:
                prepared.append({
                    'prompt': rec['prompt'],  # Will be stored as list[struct]
                    'reward_model': json.dumps(rec['reward_model']),  # Serialize to JSON string
                    'metadata': json.dumps(rec['metadata']),  # Serialize to JSON string
                    'index': rec['index']
                })
            return prepared

        train_parquet_path = self.artifact_dir / "rl_train.parquet"
        val_parquet_path = self.artifact_dir / "rl_val.parquet"

        # Prepare data
        rl_train_prepared = prepare_records_for_parquet(rl_train)
        rl_val_prepared = prepare_records_for_parquet(rl_val)

        # Create DataFrames
        rl_train_df = pd.DataFrame(rl_train_prepared)
        rl_val_df = pd.DataFrame(rl_val_prepared)

        # Define schema to ensure prompt is stored correctly
        schema = pa.schema([
            ('prompt', pa.list_(pa.struct([
                ('role', pa.string()),
                ('content', pa.string())
            ]))),
            ('reward_model', pa.string()),
            ('metadata', pa.string()),
            ('index', pa.int64())
        ])

        # Convert to PyArrow tables with schema
        train_table = pa.Table.from_pandas(rl_train_df, schema=schema)
        val_table = pa.Table.from_pandas(rl_val_df, schema=schema)

        # Write to parquet
        pq.write_table(train_table, train_parquet_path)
        pq.write_table(val_table, val_parquet_path)

        print(f"Created {len(rl_train)} RL train examples in {train_parquet_path}")
        print(f"Created {len(rl_val)} RL val examples in {val_parquet_path}")

        return rl_train_df, rl_val_df


def main():
    """CLI for running the ConnectionsManager pipeline."""
    parser = argparse.ArgumentParser(
        description="Manage Connections dataset pipeline"
    )
    parser.add_argument(
        "command",
        choices=[
            "create_dataset",
            "create_split",
            "create_generations",
            "run_sft",
            "create_rl_data",
            "run_all"
        ],
        help="Command to run"
    )
    parser.add_argument("--path", type=Path, help="Dataset path")
    parser.add_argument("--groups", type=Path, help="Path to groups.jsonl")
    parser.add_argument("--synthetic-groups", type=Path, help="Path to synthetic_groups.jsonl")
    parser.add_argument("--count-2x3", type=int, default=500, help="Number of 2x3 puzzles (default: 500)")
    parser.add_argument("--count-3x2", type=int, default=500, help="Number of 3x2 puzzles (default: 500)")
    parser.add_argument("--count-3x3", type=int, default=500, help="Number of 3x3 puzzles (default: 500)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B", help="Model name for generation")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--num-samples", type=int, help="Number of samples for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--output-dir", type=str, default="checkpoints/connections_sft", help="Output directory for SFT checkpoints")

    # Training arguments
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--sft-batch-size", type=int, default=4, help="Batch size for SFT training")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate for SFT")

    args = parser.parse_args()

    # Initialize manager
    manager = ConnectionsManager(
        path=args.path,
        groups_path=args.groups,
        synthetic_groups_path=args.synthetic_groups,
        count_2x3=args.count_2x3,
        count_3x2=args.count_3x2,
        count_3x3=args.count_3x3,
        seed=args.seed
    )

    if args.command == "create_dataset":
        manager.create_dataset()

    elif args.command == "create_split":
        manager.create_split()

    elif args.command == "create_generations":
        manager.create_generations(
            model_name=args.model,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    elif args.command == "run_sft":
        manager.run_sft(
            model_name=args.model,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.sft_batch_size,
            learning_rate=args.learning_rate,
        )

    elif args.command == "create_rl_data":
        manager.create_rl_data()

    elif args.command == "run_all":
        print("Running full pipeline...")
        manager.create_dataset()
        manager.create_split()
        print("\nNote: Generation and SFT require models. Run them separately:")
        print("  python -m data.connections.manager create_generations")
        print("  python -m data.connections.manager run_sft")
        print("  python -m data.connections.manager create_rl_data")


if __name__ == "__main__":
    main()
