"""
Standardized manager for Countdown dataset.

Manages the full pipeline from synthetic generation -> train/test splits ->
CoT generation -> SFT training -> RL data creation.

Supports multiple variants based on number of operands: 4, 5, 6, etc.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import operator
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Don't import torch or torch-dependent libraries here to avoid CUDA initialization before VLLM multiprocessing
# import torch
import pandas as pd
from datasets import load_dataset
# from trl import SFTTrainer, SFTConfig  # TRL imports torch - lazy import in run_sft instead
from vllm import LLM, SamplingParams

from data.dataset_manager import DatasetManager


# Operator definitions
OPS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': lambda a, b: a // b if b != 0 and a % b == 0 else None
}
OP_SYMBOLS = list(OPS.keys())


class ExprNode:
    """Expression tree node for countdown expressions."""

    def __init__(self, value: Union[int, str], left=None, right=None):
        self.value = value  # int or operator
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None

    def __str__(self):
        if self.is_leaf():
            return str(self.value)
        return f"({str(self.left)} {self.value} {str(self.right)})"


class CountdownManager(DatasetManager):
    """
    Manages the Countdown dataset following the standardized framework.

    Directory structure:
        data/countdown/
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
        test_split_ratio: float = 0.2,
        seed: int = 42,
        num_samples: Dict[int, int] = None,  # {num_operands: count}
        number_range: Tuple[int, int] = (1, 100),
        target_range: Tuple[int, int] = (10, 1000),
    ):
        if path is None:
            path = Path(__file__).resolve().parent
        if prompt_template is None:
            prompt_template = path / "generation_template.txt"

        super().__init__(path, prompt_template)

        self.test_split_ratio = test_split_ratio
        self.seed = seed
        self.rng = random.Random(seed)

        # Default: 500 examples each for 4, 5, 6 operands
        if num_samples is None:
            num_samples = {4: 500, 5: 500, 6: 500}
        self.num_samples = num_samples

        self.number_range = number_range
        self.target_range = target_range

    # ========================================================================
    # Dataset Creation
    # ========================================================================

    def create_dataset(self):
        """
        Creates raw_dataset.jsonl by generating countdown examples from scratch.

        Each record in raw_dataset.jsonl:
        {
            "index": <int>,
            "question": {
                "target": <int>,
                "numbers": <list of ints>,
            },
            "answer": <str>,  # Solution expression
            "metadata": {
                "variant": <str>,  # e.g., "4_operands", "5_operands"
                "num_operands": <int>,
                "hint_exprs": <list of intermediate expressions>
            }
        }
        """
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.artifact_dir / "raw_dataset.jsonl"

        print(f"Creating countdown dataset...")

        records = []
        for num_operands, count in self.num_samples.items():
            print(f"Generating {count} examples with {num_operands} operands...")
            examples = self._generate_variant(num_operands, count)
            records.extend(examples)

        # Assign sequential indices
        for idx, rec in enumerate(records):
            rec["index"] = idx

        # Write to JSONL
        with output_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"Created {len(records)} countdown examples in {output_path}")
        return records

    def _generate_variant(self, num_operands: int, count: int) -> List[dict]:
        """Generate countdown examples for a specific number of operands."""
        records = []
        attempts = 0
        max_attempts = count * 100

        while len(records) < count and attempts < max_attempts:
            attempts += 1

            # Generate a target
            target = self.rng.randint(*self.target_range)

            # Build expression tree
            tree = self._build_random_expr_tree(num_operands, target)
            if tree is None:
                continue

            # Verify the tree evaluates correctly
            result = self._evaluate_expr_tree(tree)
            if result != target:
                continue

            # Extract numbers and hints
            numbers = self._extract_leaf_numbers(tree)

            # Ensure all numbers are unique and within range
            if len(set(numbers)) != num_operands:
                continue
            if any(n < self.number_range[0] or n > self.number_range[1] for n in numbers):
                continue
            if target in numbers:  # Target shouldn't be one of the numbers
                continue

            hints = self._extract_hints(tree)
            solution_expr = str(tree)

            # Create record
            record = {
                "index": 0,  # Will be set later
                "question": {
                    "target": target,
                    "numbers": numbers,
                },
                "answer": solution_expr,
                "metadata": {
                    "variant": f"{num_operands}_operands",
                    "num_operands": num_operands,
                    "hint_exprs": hints,
                }
            }

            records.append(record)

        if len(records) < count:
            print(f"Warning: generated {len(records)}/{count} for {num_operands} operands")

        return records

    def _build_random_expr_tree(
        self,
        num_numbers: int,
        target: int,
        max_attempts: int = 100
    ) -> Optional[ExprNode]:
        """Build a random expression tree that evaluates to target."""
        if num_numbers == 1:
            return ExprNode(target)

        # Split into left and right subtrees
        split = self.rng.randint(1, num_numbers - 1)
        left_nums = split
        right_nums = num_numbers - split

        # Try different operators
        ops = OP_SYMBOLS.copy()
        self.rng.shuffle(ops)

        for op in ops:
            # Find operands that would produce target with this operator
            left_target, right_target = self._find_operands_for_target(target, op)
            if left_target is None:
                continue

            # Recursively build subtrees
            left_tree = self._build_random_expr_tree(left_nums, left_target)
            right_tree = self._build_random_expr_tree(right_nums, right_target)

            if left_tree is None or right_tree is None:
                continue

            return ExprNode(op, left_tree, right_tree)

        return None

    def _find_operands_for_target(
        self,
        target: int,
        op_symbol: str,
        max_attempts: int = 1000
    ) -> Tuple[Optional[int], Optional[int]]:
        """Find two numbers a and b such that a <op> b == target."""
        op_func = OPS[op_symbol]
        lo, hi = self.number_range

        for _ in range(max_attempts):
            if op_symbol == '+':
                a = self.rng.randint(lo, hi)
                b = target - a
            elif op_symbol == '-':
                a = self.rng.randint(lo, hi)
                b = a - target
            elif op_symbol == '*':
                if target == 0:
                    a = 0
                    b = self.rng.randint(lo, hi)
                else:
                    # Find divisors of target
                    b_candidates = [b for b in range(lo, hi + 1) if b != 0 and target % b == 0]
                    if not b_candidates:
                        continue
                    b = self.rng.choice(b_candidates)
                    a = target // b
            elif op_symbol == '/':
                b_candidates = [b for b in range(lo, hi + 1) if b != 0]
                if not b_candidates:
                    continue
                b = self.rng.choice(b_candidates)
                a = target * b
            else:
                raise ValueError(f"Unknown operator: {op_symbol}")

            # Check if both a and b are within range
            if lo <= a <= hi and lo <= b <= hi:
                # Verify to be sure
                result = op_func(a, b)
                if result == target:
                    return a, b

        return None, None

    @staticmethod
    def _extract_leaf_numbers(node: ExprNode) -> List[int]:
        """Extract all leaf numbers from the expression tree."""
        if node.is_leaf():
            return [node.value]
        return CountdownManager._extract_leaf_numbers(node.left) + \
               CountdownManager._extract_leaf_numbers(node.right)

    @staticmethod
    def _evaluate_expr_tree(node: ExprNode) -> Optional[int]:
        """Evaluate the expression tree."""
        if node.is_leaf():
            return node.value
        left_val = CountdownManager._evaluate_expr_tree(node.left)
        right_val = CountdownManager._evaluate_expr_tree(node.right)
        if left_val is None or right_val is None:
            return None
        result = OPS[node.value](left_val, right_val)
        return result

    @staticmethod
    def _extract_hints(node: ExprNode) -> List[str]:
        """Extract all intermediate expression hints (bottom-up)."""
        if node.is_leaf():
            return []

        hints = []
        hints += CountdownManager._extract_hints(node.left)
        hints += CountdownManager._extract_hints(node.right)
        hints.append(str(node))
        return hints

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
            "target": record["question"]["target"],
            "numbers": record["question"]["numbers"],
            "num_numbers": record["metadata"]["num_operands"],
        }

        # Generate prompt
        from data.prompt_loader import generate_countdown_prompt
        prompt = generate_countdown_prompt(
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

                # Check correctness
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
                all_examples = [examples_by_index[ex["index"]] for ex in examples]
                self._write_jsonl(train_path, all_examples)
                batch_count = len(batch)
                total_processed += batch_count
                print(f"Saved batch ({batch_count} new, {total_processed} total processed)")

        # Final write if output_file specified
        if output_file and not in_place:
            updated_examples = [ex for ex in examples_by_index.values() if ex.get("cot")]
            self._write_jsonl(output_file, updated_examples)
            print(f"Wrote {len(updated_examples)} generations to {output_file}")

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
                    target = record["question"]["target"]
                    numbers = record["question"]["numbers"]
                    predicted = self._extract_answer(cot)
                    return self._compare_answers(predicted, target, numbers)
        return False

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from generated text."""
        # Look for <answer> tags
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: look for Answer: pattern
        match = re.search(r"Answer:\s*(.+?)(?:\n|$)", text)
        if match:
            return match.group(1).strip()

        return None

    def _compare_answers(
        self,
        predicted: Optional[str],
        target: int,
        numbers: List[int]
    ) -> bool:
        """Compare predicted expression with target."""
        if not predicted:
            return False

        try:
            # Parse and evaluate the expression
            result = self._evaluate_expression(predicted, numbers)
            return result == target
        except Exception:
            return False

    def _evaluate_expression(self, expr: str, available_numbers: List[int]) -> Optional[int]:
        """
        Evaluate an expression and check if it uses only available numbers.

        Returns the result if valid, None otherwise.
        """
        try:
            # Remove whitespace
            expr = expr.strip()

            # Extract all numbers from the expression
            numbers_in_expr = [int(n) for n in re.findall(r'\b\d+\b', expr)]

            # Check that all numbers used are from the available list
            available_copy = available_numbers.copy()
            for num in numbers_in_expr:
                if num in available_copy:
                    available_copy.remove(num)
                else:
                    return None  # Number not available or used twice

            # Evaluate the expression
            # Create a safe evaluation environment
            allowed_names = {}
            allowed_ops = {
                '__builtins__': {},
            }

            result = eval(expr, allowed_ops, allowed_names)

            # Check if result is an integer
            if isinstance(result, (int, float)) and result == int(result):
                return int(result)

            return None
        except Exception:
            return None

    def _find_processed_indices(self, examples: List[dict]) -> set:
        """Find indices that already have CoT generations."""
        processed = set()
        for ex in examples:
            if ex.get("cot") and ex.get("cot_metadata"):
                processed.add(ex["index"])
        return processed

    def retry_failed_generations(
        self,
        model_name: str = "Qwen/Qwen2.5-3B",
        batch_size: int = 8,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.9,
        num_retries: int = 1,
        max_len: int | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Retry CoT generation for failed examples in train.jsonl.

        Identifies examples where cot_metadata.correct_answer == False
        or where the CoT is correct but exceeds max_len tokens.
        Regenerates them with potentially higher temperature for variation.

        Args:
            model_name: Model to use for generation
            batch_size: Batch size for generation
            max_new_tokens: Maximum tokens to generate
            temperature: Temperature for sampling (higher = more variation)
            top_p: Top-p for nucleus sampling
            num_retries: Number of passes through failed examples
            max_len: Maximum CoT token length; correct CoTs over this are retried (optional)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
        """
        train_path = self.artifact_dir / "train.jsonl"
        if not train_path.exists():
            raise FileNotFoundError(
                f"{train_path} not found. Run create_split() first."
            )

        print(f"Retrying failed CoT generations for {train_path}...")
        print(f"Model: {model_name}, Batch size: {batch_size}, Max tokens: {max_new_tokens}")
        print(f"Temperature: {temperature}, Num retries: {num_retries}")
        if max_len is not None:
            print(f"Max CoT length: {max_len} tokens (correct CoTs over this will be retried)")
        print(f"Tensor parallel size: {tensor_parallel_size} GPU(s)")

        # Load model with VLLM
        model = self._init_model(model_name, tensor_parallel_size, gpu_memory_utilization)

        # Load all examples
        with train_path.open("r", encoding="utf-8") as f:
            all_examples = [json.loads(line) for line in f if line.strip()]

        # Create index map for fast lookups
        examples_by_index = {ex["index"]: ex for ex in all_examples}

        # Run multiple retry passes
        for retry_num in range(num_retries):
            print(f"\n{'='*60}")
            print(f"Retry pass {retry_num + 1}/{num_retries}")
            print(f"{'='*60}")

            # Find failed examples (incorrect, missing CoT, or over max_len)
            failed_examples = []
            for ex in all_examples:
                cot_meta = ex.get("cot_metadata", {})
                is_failed = (
                    not ex.get("cot") or  # No CoT at all
                    not cot_meta or  # No metadata
                    cot_meta.get("correct_answer") is False  # Incorrect answer
                )

                # Also retry correct examples that are over max_len
                if not is_failed and max_len is not None:
                    cot_length = cot_meta.get("cot_token_length")
                    if cot_length is not None and cot_length > max_len:
                        is_failed = True

                if is_failed:
                    failed_examples.append(ex)

            if not failed_examples:
                if max_len is not None:
                    print(f"No failed examples found. All {len(all_examples)} examples are correct and under max_len!")
                else:
                    print(f"No failed examples found. All {len(all_examples)} examples are correct!")
                break

            print(f"Found {len(failed_examples)} examples to retry")

            # Count before retry
            correct_before = sum(
                1 for ex in all_examples
                if ex.get("cot_metadata", {}).get("correct_answer") is True
            )

            if max_len is not None:
                correct_and_under_len = sum(
                    1 for ex in all_examples
                    if ex.get("cot_metadata", {}).get("correct_answer") is True
                    and ex.get("cot_metadata", {}).get("cot_token_length", 0) <= max_len
                )
                print(f"Before retry: {correct_before}/{len(all_examples)} correct, "
                      f"{correct_and_under_len}/{len(all_examples)} correct & under {max_len} tokens")
            else:
                print(f"Correct examples before retry: {correct_before}/{len(all_examples)}")

            # Generate in batches
            total_improved = 0
            for i in range(0, len(failed_examples), batch_size):
                batch = failed_examples[i:i + batch_size]

                print(f"\nProcessing batch {i//batch_size + 1}/{(len(failed_examples)-1)//batch_size + 1}")
                print(f"  Indices: {[ex['index'] for ex in batch]}")

                # Generate
                prompts = [ex["prompt"] for ex in batch]
                generated_texts = self._generate_batch(
                    model, prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )

                # Process generations and update
                batch_improved = 0
                for ex, gen_text in zip(batch, generated_texts):
                    cot, cot_length = self._clean_cot(gen_text, model)

                    # Check correctness
                    is_correct = self._check_correctness(ex["index"], cot)

                    # Track improvement
                    was_correct = ex.get("cot_metadata", {}).get("correct_answer") is True
                    if is_correct and not was_correct:
                        batch_improved += 1
                        total_improved += 1

                    # Update the example in the dict
                    examples_by_index[ex["index"]]["cot"] = cot
                    examples_by_index[ex["index"]]["cot_metadata"] = {
                        "correct_answer": is_correct,
                        "cot_token_length": cot_length,
                    }

                # Write immediately after each batch
                updated_examples = [examples_by_index[ex["index"]] for ex in all_examples]
                self._write_jsonl(train_path, updated_examples)
                print(f"  Saved batch ({batch_improved} improved in this batch)")

            # Count after retry
            correct_after = sum(
                1 for ex in all_examples
                if examples_by_index[ex["index"]].get("cot_metadata", {}).get("correct_answer") is True
            )

            print(f"\nRetry pass {retry_num + 1} complete:")
            print(f"  Improved: {total_improved} examples")

            if max_len is not None:
                correct_and_under_len_after = sum(
                    1 for ex in all_examples
                    if examples_by_index[ex["index"]].get("cot_metadata", {}).get("correct_answer") is True
                    and examples_by_index[ex["index"]].get("cot_metadata", {}).get("cot_token_length", 0) <= max_len
                )
                print(f"  Correct after: {correct_after}/{len(all_examples)} ({correct_after/len(all_examples):.2%})")
                print(f"  Correct & under {max_len} tokens: {correct_and_under_len_after}/{len(all_examples)} ({correct_and_under_len_after/len(all_examples):.2%})")
            else:
                print(f"  Correct after: {correct_after}/{len(all_examples)}")
                print(f"  Accuracy: {correct_after/len(all_examples):.2%}")

            # Update all_examples for next iteration
            all_examples = [examples_by_index[ex["index"]] for ex in all_examples]

        # Final summary
        final_correct = sum(
            1 for ex in all_examples
            if examples_by_index[ex["index"]].get("cot_metadata", {}).get("correct_answer") is True
        )

        print(f"\n{'='*60}")
        print(f"Retry complete after {num_retries} pass(es)")

        if max_len is not None:
            final_correct_and_under = sum(
                1 for ex in all_examples
                if examples_by_index[ex["index"]].get("cot_metadata", {}).get("correct_answer") is True
                and examples_by_index[ex["index"]].get("cot_metadata", {}).get("cot_token_length", 0) <= max_len
            )
            print(f"Final correct: {final_correct}/{len(all_examples)} ({final_correct/len(all_examples):.2%})")
            print(f"Final correct & under {max_len} tokens: {final_correct_and_under}/{len(all_examples)} ({final_correct_and_under/len(all_examples):.2%})")
        else:
            print(f"Final accuracy: {final_correct}/{len(all_examples)} ({final_correct/len(all_examples):.2%})")

        print(f"{'='*60}")

        return [examples_by_index[ex["index"]] for ex in all_examples]

    # ========================================================================
    # SFT Training
    # ========================================================================

    def run_sft(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        output_dir: str = "checkpoints/countdown_sft",
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
        def preprocess(example):
            # Split the prompt to separate assistant prefix
            prompt = example["prompt"]
            completion = example["cot"]

            # The prompt ends with "Assistant: <think> Let me solve this step by step."
            # We want to keep everything up to "Assistant: " in prompt
            # and prepend the assistant prefix to the completion (space already in data)
            if "\n\nAssistant: " in prompt:
                parts = prompt.rsplit("\n\nAssistant: ", 1)
                prompt = parts[0] + "\n\nAssistant: "
                assistant_prefix = parts[1]  # "<think> Let me solve this step by step. " (with trailing space)
                # Don't add extra space - it's already in the assistant_prefix from the data
                completion = assistant_prefix + completion

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
                    "target": <int>,
                    "numbers": <list>,
                    "solution_expr": <str>,
                    "hint_exprs": <list>,
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
                    "target": raw_record["question"]["target"],
                    "numbers": raw_record["question"]["numbers"],
                    "solution_expr": raw_record["answer"],
                    "hint_exprs": raw_record["metadata"]["hint_exprs"],
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
        import pyarrow as pa
        import pyarrow.parquet as pq

        def prepare_records_for_parquet(records):
            """Convert records to format suitable for parquet."""
            prepared = []
            for rec in records:
                prepared.append({
                    'prompt': rec['prompt'],
                    'reward_model': json.dumps(rec['reward_model']),
                    'metadata': json.dumps(rec['metadata']),
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

        # Define schema
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
    """CLI for running the CountdownManager pipeline."""
    parser = argparse.ArgumentParser(
        description="Manage Countdown dataset pipeline"
    )
    parser.add_argument(
        "command",
        choices=[
            "create_dataset",
            "create_split",
            "create_generations",
            "retry_failed_generations",
            "run_sft",
            "create_rl_data",
            "run_all"
        ],
        help="Command to run"
    )
    parser.add_argument("--path", type=Path, help="Dataset path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B", help="Model name")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-samples", type=int, help="Number of samples for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for generation (default: 0.8)")
    parser.add_argument("--num-retries", type=int, default=1, help="Number of retry passes for failed generations (default: 1)")
    parser.add_argument("--max-len", type=int, help="Maximum CoT token length; correct CoTs over this are retried (optional)")
    parser.add_argument("--output-dir", type=str, default="checkpoints/countdown_sft", help="Output directory for SFT checkpoints")

    # Training arguments
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--sft-batch-size", type=int, default=4, help="Batch size for SFT training")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate for SFT")

    # Dataset generation parameters
    parser.add_argument("--num-4", type=int, default=500, help="Number of 4-operand examples")
    parser.add_argument("--num-5", type=int, default=500, help="Number of 5-operand examples")
    parser.add_argument("--num-6", type=int, default=500, help="Number of 6-operand examples")

    args = parser.parse_args()

    # Initialize manager
    num_samples = {}
    if args.num_4 > 0:
        num_samples[4] = args.num_4
    if args.num_5 > 0:
        num_samples[5] = args.num_5
    if args.num_6 > 0:
        num_samples[6] = args.num_6

    manager = CountdownManager(
        path=args.path,
        num_samples=num_samples if num_samples else None,
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

    elif args.command == "retry_failed_generations":
        manager.retry_failed_generations(
            model_name=args.model,
            batch_size=args.batch_size,
            temperature=args.temperature,
            num_retries=args.num_retries,
            max_len=args.max_len,
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
        print("  python -m data.countdown.manager create_generations")
        print("  python -m data.countdown.manager run_sft")
        print("  python -m data.countdown.manager create_rl_data")


if __name__ == "__main__":
    main()
