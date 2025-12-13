"""
Evaluation script for NYT Connections dataset.

Reads train.jsonl or test.jsonl, runs generation on a specified model,
and computes accuracies by variant (2x3, 3x2, 3x3) and total.

Usage:
    python -m data.connections.evaluate \\
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


class ConnectionsEvaluator:
    """Evaluator for Connections dataset."""

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
        skip_model_init: bool = False,
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

        # Initialize model only if not skipped
        if not skip_model_init:
            print(f"Loading model: {model_name}")
            print(f"Tensor parallel size: {tensor_parallel_size} GPU(s)")
            self.model = self._init_model()
        else:
            self.model = None

    def _init_model(self) -> LLM:
        """Initialize VLLM model."""
        model = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        return model

    def load_existing_results(
        self,
        output_name: str = "eval",
    ) -> Tuple[List[dict], Dict[str, float]]:
        """
        Load and display existing evaluation results.

        Args:
            output_name: Name of existing results file (results/eval_{name}.jsonl)

        Returns:
            Tuple of (results_list, accuracy_dict)
        """
        results_dir = Path("results")
        jsonl_path = results_dir / f"eval_{output_name}.jsonl"

        if not jsonl_path.exists():
            raise FileNotFoundError(f"Results file not found: {jsonl_path}")

        print(f"Loading existing results from {jsonl_path}")

        # Load results
        results = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        print(f"Loaded {len(results)} results")

        # Compute accuracies from loaded results
        accuracies = self._compute_accuracies(results)

        # Display results
        self._display_summary(accuracies, output_name)

        return results, accuracies

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
            generated_texts, finish_reasons = self._generate_batch(prompts)

            # Process each example
            for ex, gen_text, finish_reason in zip(batch, generated_texts, finish_reasons):
                idx = ex["index"]

                # Get ground truth
                if idx not in raw_data:
                    print(f"Warning: index {idx} not found in raw dataset")
                    continue

                raw_record = raw_data[idx]
                ground_truth = raw_record["answer"]
                variant = raw_record["metadata"]["variant"]

                # Check if generation was cut off due to length
                exceeded_length = finish_reason == "length"

                # Extract and compare answer
                predicted = self._extract_answer(gen_text)
                is_correct = self._compare_answers(
                    predicted,
                    ground_truth,
                    raw_record["question"]["answers"]
                )

                result = {
                    "index": idx,
                    "variant": variant,
                    "prompt": ex["prompt"],
                    "generated": gen_text,
                    "predicted_answer": predicted,
                    "ground_truth": ground_truth,
                    "correct": is_correct,
                    "exceeded_length": exceeded_length,
                    "finish_reason": finish_reason,
                }

                results.append(result)

        # Compute accuracies
        accuracies = self._compute_accuracies(results)

        # Save results
        self._save_results(results, accuracies, output_name)

        return results, accuracies

    def _generate_batch(self, prompts: List[str]) -> Tuple[List[str], List[str]]:
        """Generate text for a batch of prompts.

        Returns:
            Tuple of (generated_texts, finish_reasons)
        """
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )

        outputs = self.model.generate(prompts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
        finish_reasons = [output.outputs[0].finish_reason for output in outputs]

        return generated_texts, finish_reasons

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
        return re.sub(r"\s+", "", answer.lower())

    @staticmethod
    def _parse_answer_groups(answer: str) -> List[List[str]]:
        """Parse answer into groups of words."""
        groups = []
        pattern = r"\[(.*?)\]"
        matches = re.findall(pattern, answer, re.DOTALL)
        for match in matches:
            words = [w.strip().upper() for w in match.split(",")]
            groups.append(sorted(words))
        return groups

    def _compute_accuracies(self, results: List[dict]) -> Dict[str, float]:
        """Compute accuracies by variant and total."""
        # Count by variant
        variant_counts = {}
        variant_correct = {}
        variant_counts_excl_length = {}
        variant_correct_excl_length = {}
        variant_incorrect_due_to_length = {}
        variant_total_incorrect = {}

        for result in results:
            variant = result["variant"]
            if variant not in variant_counts:
                variant_counts[variant] = 0
                variant_correct[variant] = 0
                variant_counts_excl_length[variant] = 0
                variant_correct_excl_length[variant] = 0
                variant_incorrect_due_to_length[variant] = 0
                variant_total_incorrect[variant] = 0

            variant_counts[variant] += 1
            exceeded_length = result.get("exceeded_length", False)

            if result["correct"]:
                variant_correct[variant] += 1
                # Also count in excl_length if not exceeded
                if not exceeded_length:
                    variant_counts_excl_length[variant] += 1
                    variant_correct_excl_length[variant] += 1
            else:
                variant_total_incorrect[variant] += 1
                if exceeded_length:
                    variant_incorrect_due_to_length[variant] += 1
                else:
                    # Count incorrect that didn't exceed length
                    variant_counts_excl_length[variant] += 1

        # Compute accuracies
        accuracies = {}
        for variant in variant_counts:
            # Regular accuracy
            acc = variant_correct[variant] / variant_counts[variant] if variant_counts[variant] > 0 else 0.0
            accuracies[variant] = acc

            # Accuracy excluding length-exceeded examples
            acc_excl = (
                variant_correct_excl_length[variant] / variant_counts_excl_length[variant]
                if variant_counts_excl_length[variant] > 0 else 0.0
            )
            accuracies[f"{variant}_excl_length"] = acc_excl

            # Proportion of incorrect due to length
            prop_length = (
                variant_incorrect_due_to_length[variant] / variant_total_incorrect[variant]
                if variant_total_incorrect[variant] > 0 else 0.0
            )
            accuracies[f"{variant}_prop_length"] = prop_length

        # Total accuracy
        total_correct = sum(variant_correct.values())
        total_count = sum(variant_counts.values())
        accuracies["total"] = total_correct / total_count if total_count > 0 else 0.0

        # Total accuracy excluding length
        total_correct_excl = sum(variant_correct_excl_length.values())
        total_count_excl = sum(variant_counts_excl_length.values())
        accuracies["total_excl_length"] = total_correct_excl / total_count_excl if total_count_excl > 0 else 0.0

        # Total proportion of incorrect due to length
        total_incorrect_length = sum(variant_incorrect_due_to_length.values())
        total_incorrect = sum(variant_total_incorrect.values())
        accuracies["total_prop_length"] = total_incorrect_length / total_incorrect if total_incorrect > 0 else 0.0

        # Add counts
        accuracies["counts"] = variant_counts
        accuracies["correct"] = variant_correct
        accuracies["counts_excl_length"] = variant_counts_excl_length
        accuracies["correct_excl_length"] = variant_correct_excl_length
        accuracies["incorrect_due_to_length"] = variant_incorrect_due_to_length
        accuracies["total_incorrect"] = variant_total_incorrect

        return accuracies

    def _display_summary(self, accuracies: Dict[str, float], output_name: str):
        """Display evaluation summary."""
        # Prepare CSV data
        csv_data = []
        for variant in ["2x3", "3x2", "3x3"]:
            if variant in accuracies:
                csv_data.append({
                    "variant": variant,
                    "accuracy": accuracies[variant],
                    "accuracy_excl_length": accuracies[f"{variant}_excl_length"],
                    "prop_incorrect_due_to_length": accuracies[f"{variant}_prop_length"],
                    "correct": accuracies["correct"].get(variant, 0),
                    "total": accuracies["counts"].get(variant, 0),
                    "correct_excl_length": accuracies["correct_excl_length"].get(variant, 0),
                    "total_excl_length": accuracies["counts_excl_length"].get(variant, 0),
                    "incorrect_due_to_length": accuracies["incorrect_due_to_length"].get(variant, 0),
                    "total_incorrect": accuracies["total_incorrect"].get(variant, 0),
                })

        # Add total
        csv_data.append({
            "variant": "total",
            "accuracy": accuracies["total"],
            "accuracy_excl_length": accuracies["total_excl_length"],
            "prop_incorrect_due_to_length": accuracies["total_prop_length"],
            "correct": sum(accuracies["correct"].values()),
            "total": sum(accuracies["counts"].values()),
            "correct_excl_length": sum(accuracies["correct_excl_length"].values()),
            "total_excl_length": sum(accuracies["counts_excl_length"].values()),
            "incorrect_due_to_length": sum(accuracies["incorrect_due_to_length"].values()),
            "total_incorrect": sum(accuracies["total_incorrect"].values()),
        })

        # Print summary
        print("\n=== Evaluation Results ===")
        print(f"Model: {self.model_name}")
        print(f"Output name: {output_name}")
        print("\nAccuracies:")
        for row in csv_data:
            print(f"  {row['variant']:5s}:")
            print(f"    Accuracy:                    {row['accuracy']:.2%} ({row['correct']}/{row['total']})")
            print(f"    Accuracy (excl. length):     {row['accuracy_excl_length']:.2%} ({row['correct_excl_length']}/{row['total_excl_length']})")
            print(f"    Incorrect due to length:     {row['prop_incorrect_due_to_length']:.2%} ({row['incorrect_due_to_length']}/{row['total_incorrect']})")

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
        for variant in ["2x3", "3x2", "3x3"]:
            if variant in accuracies:
                csv_data.append({
                    "variant": variant,
                    "accuracy": accuracies[variant],
                    "accuracy_excl_length": accuracies[f"{variant}_excl_length"],
                    "prop_incorrect_due_to_length": accuracies[f"{variant}_prop_length"],
                    "correct": accuracies["correct"].get(variant, 0),
                    "total": accuracies["counts"].get(variant, 0),
                    "correct_excl_length": accuracies["correct_excl_length"].get(variant, 0),
                    "total_excl_length": accuracies["counts_excl_length"].get(variant, 0),
                    "incorrect_due_to_length": accuracies["incorrect_due_to_length"].get(variant, 0),
                    "total_incorrect": accuracies["total_incorrect"].get(variant, 0),
                })

        # Add total
        csv_data.append({
            "variant": "total",
            "accuracy": accuracies["total"],
            "accuracy_excl_length": accuracies["total_excl_length"],
            "prop_incorrect_due_to_length": accuracies["total_prop_length"],
            "correct": sum(accuracies["correct"].values()),
            "total": sum(accuracies["counts"].values()),
            "correct_excl_length": sum(accuracies["correct_excl_length"].values()),
            "total_excl_length": sum(accuracies["counts_excl_length"].values()),
            "incorrect_due_to_length": sum(accuracies["incorrect_due_to_length"].values()),
            "total_incorrect": sum(accuracies["total_incorrect"].values()),
        })

        # Write CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)

        print(f"Saved accuracy summary to {csv_path}")

        # Display summary
        self._display_summary(accuracies, output_name)


def main():
    """CLI for running evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate model on Connections dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name for generation (e.g., Qwen/Qwen2.5-3B)"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for output files (results/eval_{name}.jsonl)"
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Load and display existing results instead of running evaluation"
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
        help="Data directory containing artifacts/ (default: data/connections/artifacts)"
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

    # If use-existing flag is set, load existing results
    if args.use_existing:
        # Create a minimal evaluator just for loading results (no model needed)
        evaluator = ConnectionsEvaluator(
            data_dir=args.data_dir,
            model_name=args.model or "dummy",  # Model not needed for loading
            batch_size=1,
            skip_model_init=True,  # Skip model initialization when loading existing results
        )
        evaluator.load_existing_results(output_name=args.name)
        return

    # Validate model is provided for new evaluation
    if not args.model:
        parser.error("--model is required unless --use-existing is specified")

    # Initialize evaluator
    evaluator = ConnectionsEvaluator(
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
