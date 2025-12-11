"""
Evaluation script for Countdown dataset.

Reads train.jsonl or test.jsonl, runs generation on a specified model,
and computes accuracies by variant (4_operands, 5_operands, 6_operands) and total.

Usage:
    python -m data.countdown.evaluate \\
        --model Qwen/Qwen2.5-3B \\
        --name baseline_3b \\
        --input test \\
        --batch-size 16
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from vllm import LLM, SamplingParams


class CountdownEvaluator:
    """Evaluator for Countdown dataset."""

    def __init__(
        self,
        data_dir: Path = None,
        model_name: str = "Qwen/Qwen2.5-3B",
        batch_size: int = 16,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent / "artifacts"

        self.data_dir = data_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization

        # Initialize model
        print(f"Loading model: {model_name}")
        print(f"Tensor parallel size: {tensor_parallel_size} GPU(s)")
        self.model = self._init_model()

    def _init_model(self) -> LLM:
        """Initialize VLLM model."""
        model = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        return model

    def evaluate(
        self,
        input_file: str = "test",
        output_name: str = "eval",
    ) -> Tuple[List[dict], Dict[str, float]]:
        """
        Run evaluation on the specified input file.

        Args:
            input_file: Either "train", "test", or a full path to a JSONL file
            output_name: Name for output files (results/eval_{name}.jsonl)

        Returns:
            Tuple of (results_list, accuracy_dict)
        """
        # Determine input path
        if input_file in ["train", "test"]:
            input_path = self.data_dir / f"{input_file}.jsonl"
        else:
            input_path = Path(input_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        print(f"Evaluating on {input_path}")

        # Load examples
        examples = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        print(f"Loaded {len(examples)} examples")

        # Load raw dataset for ground truth
        raw_path = self.data_dir / "raw_dataset.jsonl"
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

        raw_data = {}
        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                raw_data[record["index"]] = record

        # Run generation in batches
        results = []
        for i in range(0, len(examples), self.batch_size):
            batch = examples[i:i + self.batch_size]
            print(f"Processing batch {i//self.batch_size + 1}/{(len(examples)-1)//self.batch_size + 1}")

            # Generate
            prompts = [ex["prompt"] for ex in batch]
            generated_texts = self._generate_batch(prompts)

            # Process each example
            for ex, gen_text in zip(batch, generated_texts):
                idx = ex["index"]

                # Get ground truth
                if idx not in raw_data:
                    print(f"Warning: index {idx} not found in raw dataset")
                    continue

                raw_record = raw_data[idx]
                target = raw_record["question"]["target"]
                numbers = raw_record["question"]["numbers"]
                variant = raw_record["metadata"]["variant"]
                solution_expr = raw_record["answer"]

                # Extract and compare answer
                predicted = self._extract_answer(gen_text)
                is_correct = self._compare_answers(predicted, target, numbers)

                result = {
                    "index": idx,
                    "variant": variant,
                    "target": target,
                    "numbers": numbers,
                    "prompt": ex["prompt"],
                    "generated": gen_text,
                    "predicted_answer": predicted,
                    "ground_truth": solution_expr,
                    "correct": is_correct,
                }

                results.append(result)

        # Compute accuracies
        accuracies = self._compute_accuracies(results)

        # Save results
        self._save_results(results, accuracies, output_name)

        return results, accuracies

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate text for a batch of prompts."""
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )

        outputs = self.model.generate(prompts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]

        return generated_texts

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

    def _compute_accuracies(self, results: List[dict]) -> Dict[str, float]:
        """Compute accuracies by variant and total."""
        # Count by variant
        variant_counts = {}
        variant_correct = {}

        for result in results:
            variant = result["variant"]
            if variant not in variant_counts:
                variant_counts[variant] = 0
                variant_correct[variant] = 0

            variant_counts[variant] += 1
            if result["correct"]:
                variant_correct[variant] += 1

        # Compute accuracies
        accuracies = {}
        for variant in variant_counts:
            acc = variant_correct[variant] / variant_counts[variant] if variant_counts[variant] > 0 else 0.0
            accuracies[variant] = acc

        # Total accuracy
        total_correct = sum(variant_correct.values())
        total_count = sum(variant_counts.values())
        accuracies["total"] = total_correct / total_count if total_count > 0 else 0.0

        # Add counts
        accuracies["counts"] = variant_counts
        accuracies["correct"] = variant_correct

        return accuracies

    def _save_results(
        self,
        results: List[dict],
        accuracies: Dict[str, float],
        output_name: str
    ):
        """Save results to JSONL and CSV."""
        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Save JSONL
        jsonl_path = results_dir / f"eval_{output_name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"\nSaved results to {jsonl_path}")

        # Save CSV with accuracies
        csv_path = results_dir / f"eval_{output_name}.csv"

        # Prepare CSV data
        csv_data = []
        for variant in sorted(accuracies.get("counts", {}).keys()):
            csv_data.append({
                "variant": variant,
                "accuracy": accuracies[variant],
                "correct": accuracies["correct"].get(variant, 0),
                "total": accuracies["counts"].get(variant, 0),
            })

        # Add total
        csv_data.append({
            "variant": "total",
            "accuracy": accuracies["total"],
            "correct": sum(accuracies["correct"].values()),
            "total": sum(accuracies["counts"].values()),
        })

        # Write CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)

        print(f"Saved accuracy summary to {csv_path}")

        # Print summary
        print("\n=== Evaluation Results ===")
        print(f"Model: {self.model_name}")
        print(f"Output name: {output_name}")
        print("\nAccuracies:")
        for row in csv_data:
            print(f"  {row['variant']:12s}: {row['accuracy']:.2%} ({row['correct']}/{row['total']})")


def main():
    """CLI for running evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate model on Countdown dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name for generation (e.g., Qwen/Qwen2.5-3B)"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for output files (results/eval_{name}.jsonl)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="test",
        help="Input file: 'train', 'test', or path to JSONL (default: test)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Data directory containing artifacts/ (default: data/countdown/artifacts)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for generation (default: 16)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max new tokens for generation (default: 2048)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p for generation (default: 1.0)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = CountdownEvaluator(
        data_dir=args.data_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Run evaluation
    evaluator.evaluate(
        input_file=args.input,
        output_name=args.name,
    )


if __name__ == "__main__":
    main()
