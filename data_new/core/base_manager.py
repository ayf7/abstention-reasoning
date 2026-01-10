"""
Abstract base class for dataset managers with shared implementation.

Subclasses must implement task-specific hooks:
    - create_dataset(): Generate raw_dataset.jsonl
    - _build_prompt_messages(): Convert raw record to messages list
    - _extract_answer(): Extract answer from CoT text
    - _compare_answers(): Compare predicted vs ground truth
    - _get_reward_model_data(): Build reward_model dict for RL

Base class provides shared implementation for:
    - JSONL/Parquet I/O with proper indexing
    - VLLM model initialization and batch generation
    - CoT cleaning and token counting
    - Correctness checking with O(1) index lookup
    - SFT training via TRL
    - Train/test splitting
    - RL data parquet creation
    - RNG state persistence for reproducibility
"""
from __future__ import annotations

import pickle
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from data_new.core.io_utils import build_index, load_jsonl, write_jsonl, write_parquet
from data_new.core.vllm_generator import VLLMGenerator

if TYPE_CHECKING:
    from omegaconf import DictConfig


# Type alias for message format
Message = Dict[str, str]  # {"role": str, "content": str}


class BaseManager(ABC):
    """
    Abstract base class for dataset managers.

    Provides shared implementation for the data pipeline while requiring
    task-specific implementations for dataset creation and answer evaluation.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize manager with Hydra config.

        Args:
            cfg: OmegaConf config with sections:
                - task: {name, seed, test_split_ratio}
                - paths: {root_dir, artifacts_dir, template_path}
                - generation: VLLM generation settings
                - sft: SFT training settings
                - rl: RL data settings
                - prompt: {system_message, assistant_prefix}
        """
        self.cfg = cfg
        self.task_name = cfg.task.name
        self.artifact_dir = Path(cfg.paths.artifacts_dir)
        self.template_path = Path(cfg.paths.template_path)
        self.seed = cfg.task.seed

        # Initialize RNG - restore state if exists, otherwise use seed
        self.rng = random.Random(self.seed)
        self._load_rng_state()

        # Lazy-loaded index for O(1) correctness checks
        self._raw_index: Optional[Dict[int, Dict]] = None

    # ========================================================================
    # File path helpers (task-prefixed filenames)
    # ========================================================================

    def _artifact_path(self, filename: str) -> Path:
        """Get path to artifact file with task prefix."""
        return self.artifact_dir / f"{self.task_name}_{filename}"

    @property
    def raw_dataset_path(self) -> Path:
        return self._artifact_path("raw_dataset.jsonl")

    @property
    def train_path(self) -> Path:
        return self._artifact_path("train.jsonl")

    @property
    def test_path(self) -> Path:
        return self._artifact_path("test.jsonl")

    @property
    def rl_train_path(self) -> Path:
        return self._artifact_path("rl_train.parquet")

    @property
    def rl_val_path(self) -> Path:
        return self._artifact_path("rl_val.parquet")

    @property
    def rng_state_path(self) -> Path:
        return self._artifact_path("rng_state.pkl")

    # ========================================================================
    # RNG state persistence for reproducibility
    # ========================================================================

    def _save_rng_state(self) -> None:
        """Save RNG state for reproducibility across commands."""
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        with self.rng_state_path.open("wb") as f:
            pickle.dump(self.rng.getstate(), f)

    def _load_rng_state(self) -> None:
        """Load RNG state if it exists, otherwise keep seed-initialized state."""
        if self.rng_state_path.exists():
            with self.rng_state_path.open("rb") as f:
                state = pickle.load(f)
                self.rng.setstate(state)
            print(f"Restored RNG state from {self.rng_state_path}")

    # ========================================================================
    # Abstract methods (task-specific implementations required)
    # ========================================================================

    @abstractmethod
    def create_dataset(self) -> List[Dict]:
        """
        Create raw_dataset.jsonl with task-specific generation logic.

        Each record should have:
            - index: int
            - question: dict (task-specific structure)
            - answer: str
            - metadata: dict

        Returns:
            List of generated records
        """
        pass

    @abstractmethod
    def _build_prompt_messages(self, raw_record: Dict) -> List[Message]:
        """
        Convert raw record to messages list format.

        Args:
            raw_record: Record from raw_dataset.jsonl

        Returns:
            List of messages: [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}  # prefix
            ]
        """
        pass

    @abstractmethod
    def _extract_answer(self, cot_text: str) -> Optional[str]:
        """
        Extract answer from generated CoT text.

        Args:
            cot_text: Generated chain-of-thought text

        Returns:
            Extracted answer string, or None if not found
        """
        pass

    @abstractmethod
    def _compare_answers(self, predicted: Optional[str], raw_record: Dict) -> bool:
        """
        Compare predicted answer with ground truth from raw record.

        Args:
            predicted: Extracted answer from CoT
            raw_record: Original record with ground truth

        Returns:
            True if answer is correct
        """
        pass

    @abstractmethod
    def _get_reward_model_data(self, raw_record: Dict) -> Dict[str, Any]:
        """
        Build reward_model dict for RL training.

        Args:
            raw_record: Original record from raw_dataset.jsonl

        Returns:
            Dict with "ground_truth" key containing task-specific data
        """
        pass

    # ========================================================================
    # Shared implementations
    # ========================================================================

    def create_split(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Create train.jsonl and test.jsonl from raw_dataset.jsonl.

        Each record in train/test:
            - index: int
            - prompt: List[Message]
            - cot: str (initially empty)
            - cot_metadata: dict (initially empty)

        Returns:
            (train_records, test_records)
        """
        if not self.raw_dataset_path.exists():
            raise FileNotFoundError(f"{self.raw_dataset_path} not found. Run create_dataset() first.")

        print(f"Creating train/test split from {self.raw_dataset_path}...")

        records = load_jsonl(self.raw_dataset_path)
        self.rng.shuffle(records)

        test_size = int(len(records) * self.cfg.task.test_split_ratio)
        test_records = [self._to_split_format(r) for r in records[:test_size]]
        train_records = [self._to_split_format(r) for r in records[test_size:]]

        # Sort by index for deterministic ordering
        train_records.sort(key=lambda x: x["index"])
        test_records.sort(key=lambda x: x["index"])

        write_jsonl(self.train_path, train_records)
        write_jsonl(self.test_path, test_records)

        # Save RNG state for reproducibility
        self._save_rng_state()

        print(f"Created {len(train_records)} train examples in {self.train_path}")
        print(f"Created {len(test_records)} test examples in {self.test_path}")

        return train_records, test_records

    def _to_split_format(self, raw_record: Dict) -> Dict:
        """Convert raw record to split format with messages prompt."""
        return {
            "index": raw_record["index"],
            "prompt": self._build_prompt_messages(raw_record),
            "cot": "",
            "cot_metadata": {},
        }

    def _ensure_raw_index(self) -> None:
        """Build or return cached raw_dataset index for O(1) lookups."""
        if self._raw_index is None:
            records = load_jsonl(self.raw_dataset_path)
            self._raw_index = build_index(records, key="index")

    def _check_correctness_fast(self, index: int, cot: str) -> bool:
        """O(1) correctness check using pre-built index."""
        self._ensure_raw_index()
        raw_record = self._raw_index.get(index)
        if raw_record is None:
            return False
        predicted = self._extract_answer(cot)
        return self._compare_answers(predicted, raw_record)

    def _messages_to_string(self, messages: List[Message]) -> str:
        """
        Convert messages list to string for VLLM generation.

        Format: "System: ...\n\nUser: ...\n\nAssistant: ..."
        """
        parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            parts.append(f"{role}: {msg['content']}")
        return "\n\n".join(parts)

    def _find_processed_indices(self, examples: List[Dict], retry_incorrect: bool = False) -> set:
        """Find indices that already have CoT generations.

        Args:
            examples: List of examples to check
            retry_incorrect: If True, don't skip incorrect examples (they'll be retried)
        """
        processed = set()
        for ex in examples:
            if ex.get("cot") and ex.get("cot_metadata"):
                if retry_incorrect and not ex["cot_metadata"].get("correct_answer"):
                    # Don't mark as processed - will be retried
                    continue
                processed.add(ex["index"])
        return processed

    def create_generations(
        self,
        in_place: bool = True,
        output_file: Path | None = None,
    ) -> List[Dict]:
        """
        Generate CoTs for train.jsonl with resumption support.

        Args:
            in_place: If True, update train.jsonl after each batch
            output_file: If set, write final results here (only if not in_place)

        Config options:
            generation.retry_incorrect: If True, retry incorrect examples

        Returns:
            List of examples with CoT generations
        """
        if not self.train_path.exists():
            raise FileNotFoundError(f"{self.train_path} not found. Run create_split() first.")

        retry_incorrect = getattr(self.cfg.generation, "retry_incorrect", False)

        print(f"Generating CoTs for {self.train_path}...")
        if retry_incorrect:
            print("Mode: retry_incorrect=True (will retry incorrect examples)")

        # Build index for O(1) correctness checks
        self._ensure_raw_index()

        # Load examples
        examples = load_jsonl(self.train_path)

        # Initialize generator
        generator = VLLMGenerator(self.cfg.generation)

        processed_indices = self._find_processed_indices(examples, retry_incorrect) if in_place else set()
        examples_by_index = {ex["index"]: ex for ex in examples}

        print(f"Processing {len(examples)} examples...")
        if processed_indices:
            print(f"Resuming: {len(processed_indices)} already processed")

        total_processed = len(processed_indices)

        for batch in generator.batch_iter(examples):
            # Filter already processed
            batch = [ex for ex in batch if ex["index"] not in processed_indices]
            if not batch:
                continue

            print(f"Processing batch: indices {[ex['index'] for ex in batch]}")

            # Convert messages to string for VLLM
            prompts = [self._messages_to_string(ex["prompt"]) for ex in batch]
            generated = generator.generate(prompts)

            for ex, text in zip(batch, generated):
                cot, length = generator.clean_cot(text)
                is_correct = self._check_correctness_fast(ex["index"], cot)

                examples_by_index[ex["index"]]["cot"] = cot
                examples_by_index[ex["index"]]["cot_metadata"] = {
                    "correct_answer": is_correct,
                    "cot_token_length": length,
                }
                processed_indices.add(ex["index"])

            if in_place:
                all_examples = [examples_by_index[ex["index"]] for ex in examples]
                write_jsonl(self.train_path, all_examples)
                total_processed += len(batch)
                print(f"Saved batch ({len(batch)} new, {total_processed} total)")

        if output_file and not in_place:
            updated = [ex for ex in examples_by_index.values() if ex.get("cot")]
            write_jsonl(output_file, updated)
            print(f"Wrote {len(updated)} generations to {output_file}")

        return [ex for ex in examples_by_index.values() if ex.get("cot")]

    def retry_failed_generations(self) -> List[Dict]:
        """
        Retry CoT generation for failed examples.

        Uses retry config: {num_retries, temperature, max_len}

        Returns:
            Updated list of all examples
        """
        if not self.train_path.exists():
            raise FileNotFoundError(f"{self.train_path} not found. Run create_split() first.")

        retry_cfg = self.cfg.retry
        print(f"Retrying failed CoT generations...")
        print(f"Num retries: {retry_cfg.num_retries}, Temperature: {retry_cfg.temperature}")
        if retry_cfg.max_len:
            print(f"Max CoT length: {retry_cfg.max_len} tokens")

        self._ensure_raw_index()
        examples = load_jsonl(self.train_path)
        examples_by_index = {ex["index"]: ex for ex in examples}

        # Override temperature for retries
        gen_cfg = self.cfg.generation.copy()
        gen_cfg.temperature = retry_cfg.temperature
        generator = VLLMGenerator(gen_cfg)

        for retry_num in range(retry_cfg.num_retries):
            print(f"\n{'='*60}")
            print(f"Retry pass {retry_num + 1}/{retry_cfg.num_retries}")

            # Find failed examples
            failed = []
            for ex in examples:
                meta = ex.get("cot_metadata", {})
                is_failed = (
                    not ex.get("cot") or
                    not meta or
                    meta.get("correct_answer") is False
                )
                # Also retry if over max_len
                if not is_failed and retry_cfg.max_len:
                    if meta.get("cot_token_length", 0) > retry_cfg.max_len:
                        is_failed = True
                if is_failed:
                    failed.append(ex)

            if not failed:
                print("No failed examples remaining!")
                break

            print(f"Found {len(failed)} examples to retry")

            for batch in generator.batch_iter(failed):
                prompts = [self._messages_to_string(ex["prompt"]) for ex in batch]
                generated = generator.generate(prompts)

                for ex, text in zip(batch, generated):
                    cot, length = generator.clean_cot(text)
                    is_correct = self._check_correctness_fast(ex["index"], cot)

                    examples_by_index[ex["index"]]["cot"] = cot
                    examples_by_index[ex["index"]]["cot_metadata"] = {
                        "correct_answer": is_correct,
                        "cot_token_length": length,
                    }

                # Save after each batch
                all_examples = [examples_by_index[ex["index"]] for ex in examples]
                write_jsonl(self.train_path, all_examples)

            # Update examples for next iteration
            examples = [examples_by_index[ex["index"]] for ex in examples]

        # Final stats
        correct = sum(1 for ex in examples if ex.get("cot_metadata", {}).get("correct_answer"))
        print(f"\nFinal: {correct}/{len(examples)} correct ({100*correct/len(examples):.1f}%)")

        return examples

    def run_sft(self) -> Any:
        """
        Run SFT training on correct CoT examples.

        Uses sft config: {model_name, output_dir, num_epochs, batch_size, learning_rate, ...}

        Returns:
            SFTTrainer instance
        """
        from data_new.core.sft_trainer import run_sft_training

        def filter_correct(example):
            meta = example.get("cot_metadata", {})
            return (
                meta.get("correct_answer") is True
                and isinstance(meta.get("cot_token_length"), int)
                and example.get("cot", "").strip()
            )

        def preprocess_for_sft(example):
            """Convert messages prompt + cot to single text field."""
            messages = example["prompt"]
            cot = example["cot"]

            # Build full text from messages + CoT completion
            text_parts = []
            for msg in messages:
                role = msg["role"].capitalize()
                text_parts.append(f"{role}: {msg['content']}")

            # Add CoT as continuation of assistant message
            text_parts[-1] = text_parts[-1] + cot

            return {
                "text": "\n\n".join(text_parts),
            }

        return run_sft_training(
            train_path=self.train_path,
            cfg=self.cfg.sft,
            filter_fn=filter_correct,
            preprocess_fn=preprocess_for_sft,
            seed=self.seed,
        )

    def reevaluate_correctness(self) -> List[Dict]:
        """
        Re-evaluate correctness for all examples in train.jsonl.

        Useful after fixing data issues (e.g., character normalization).
        Updates cot_metadata.correct_answer in place.

        Returns:
            Updated list of examples
        """
        if not self.train_path.exists():
            raise FileNotFoundError(f"{self.train_path} not found.")

        print(f"Re-evaluating correctness for {self.train_path}...")

        self._ensure_raw_index()
        examples = load_jsonl(self.train_path)

        old_correct = sum(1 for ex in examples if ex.get("cot_metadata", {}).get("correct_answer"))

        for ex in examples:
            cot = ex.get("cot", "")
            if not cot:
                continue
            is_correct = self._check_correctness_fast(ex["index"], cot)
            ex["cot_metadata"]["correct_answer"] = is_correct

        new_correct = sum(1 for ex in examples if ex.get("cot_metadata", {}).get("correct_answer"))

        write_jsonl(self.train_path, examples)

        print(f"Before: {old_correct} correct")
        print(f"After:  {new_correct} correct")
        print(f"Change: {new_correct - old_correct:+d}")

        return examples

    def analyze_generations(self) -> Dict[str, Any]:
        """
        Analyze CoT generation statistics from train.jsonl.

        Prints:
            - Correct answers: min, 25p, median, 75p, max token lengths
            - Incorrect answers: fraction below 2048, fraction incomplete

        Returns:
            Dict with stats: {correct: {...}, incorrect: {...}, summary: {...}}
        """
        if not self.train_path.exists():
            raise FileNotFoundError(f"{self.train_path} not found. Run create_generations() first.")

        examples = load_jsonl(self.train_path)

        correct_lengths = []
        incorrect_lengths = []
        incomplete_indices = []
        complete_but_wrong_indices = []

        for ex in examples:
            meta = ex.get("cot_metadata", {})
            if not meta:
                continue

            length = meta.get("cot_token_length", 0)
            cot = ex.get("cot", "")
            idx = ex.get("index")

            if meta.get("correct_answer"):
                correct_lengths.append(length)
            else:
                incorrect_lengths.append(length)
                # Incomplete = missing </think> or <answer> tags (hit token limit)
                is_incomplete = "</think>" not in cot or "<answer>" not in cot
                if is_incomplete:
                    incomplete_indices.append(idx)
                else:
                    complete_but_wrong_indices.append(idx)

        if not correct_lengths and not incorrect_lengths:
            print("No generations found in train.jsonl")
            return {}

        # Compute stats
        import numpy as np
        correct_arr = np.array(correct_lengths) if correct_lengths else np.array([0])
        incorrect_arr = np.array(incorrect_lengths) if incorrect_lengths else np.array([0])

        num_incomplete = len(incomplete_indices)
        num_complete_but_wrong = len(complete_but_wrong_indices)

        stats = {
            "summary": {
                "total": len(correct_lengths) + len(incorrect_lengths),
                "correct": len(correct_lengths),
                "incorrect": len(incorrect_lengths),
                "accuracy": len(correct_lengths) / (len(correct_lengths) + len(incorrect_lengths)) * 100,
            },
            "correct": {
                "min": int(np.min(correct_arr)),
                "p25": int(np.percentile(correct_arr, 25)),
                "median": int(np.percentile(correct_arr, 50)),
                "p75": int(np.percentile(correct_arr, 75)),
                "max": int(np.max(correct_arr)),
            } if correct_lengths else None,
            "incorrect": {
                "count": len(incorrect_lengths),
                "incomplete": num_incomplete,
                "incomplete_pct": num_incomplete / len(incorrect_lengths) * 100 if incorrect_lengths else 0,
                "incomplete_indices": incomplete_indices,
                "complete_but_wrong": num_complete_but_wrong,
                "complete_but_wrong_pct": num_complete_but_wrong / len(incorrect_lengths) * 100 if incorrect_lengths else 0,
                "complete_but_wrong_indices": complete_but_wrong_indices,
            } if incorrect_lengths else None,
        }

        # Print report
        print("\n" + "=" * 60)
        print("GENERATION ANALYSIS")
        print("=" * 60)
        print(f"Total samples: {stats['summary']['total']}")
        print(f"Correct: {stats['summary']['correct']} ({stats['summary']['accuracy']:.1f}%)")
        print(f"Incorrect: {stats['summary']['incorrect']}")

        if stats["correct"]:
            print("\nCORRECT ANSWERS - Token Lengths:")
            print(f"  Min:    {stats['correct']['min']:,}")
            print(f"  25p:    {stats['correct']['p25']:,}")
            print(f"  Median: {stats['correct']['median']:,}")
            print(f"  75p:    {stats['correct']['p75']:,}")
            print(f"  Max:    {stats['correct']['max']:,}")

        if stats["incorrect"]:
            print("\nINCORRECT ANSWERS:")
            print(f"  Incomplete (hit token limit): {stats['incorrect']['incomplete']} ({stats['incorrect']['incomplete_pct']:.1f}%)")
            if incomplete_indices:
                sample = incomplete_indices[:10]
                print(f"    Example indices: {sample}")
            print(f"  Complete but wrong: {stats['incorrect']['complete_but_wrong']} ({stats['incorrect']['complete_but_wrong_pct']:.1f}%)")
            if complete_but_wrong_indices:
                sample = complete_but_wrong_indices[:10]
                print(f"    Example indices: {sample}")

        print("=" * 60 + "\n")

        return stats

    def create_rl_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Create rl_train.parquet and rl_val.parquet from train.jsonl.

        Format is compatible with verl's RLHFDataset:
            - data_source: str (task name for reward function selection)
            - prompt: List[Message] (chat format)
            - reward_model: {"style": "rule", "ground_truth": {...}}
            - extra_info: dict (metadata)

        Returns:
            (rl_train_records, rl_val_records)
        """
        if not self.train_path.exists():
            raise FileNotFoundError(f"{self.train_path} not found. Run create_split() first.")

        print(f"Creating RL data from {self.train_path}...")

        self._ensure_raw_index()
        examples = load_jsonl(self.train_path)

        rl_records = []
        for ex in examples:
            idx = ex["index"]
            raw = self._raw_index.get(idx)
            if raw is None:
                continue

            # Build verl-compatible record
            reward_data = self._get_reward_model_data(raw)
            rl_records.append({
                "data_source": self.cfg.task.name,  # For reward function selection
                "prompt": ex["prompt"],  # Already messages list
                "reward_model": {
                    "style": "rule",
                    "ground_truth": reward_data.get("ground_truth", reward_data),
                },
                "extra_info": {
                    "index": idx,
                    **raw.get("metadata", {}),
                },
            })

        # Shuffle and split
        self.rng.shuffle(rl_records)
        split = int(len(rl_records) * self.cfg.rl.train_split_ratio)
        rl_train = rl_records[:split]
        rl_val = rl_records[split:]

        # Write parquet using HuggingFace datasets - keeps nested structures native
        # (verl expects prompt and reward_model as native dicts/lists, not JSON strings)
        write_parquet(self.rl_train_path, rl_train)
        write_parquet(self.rl_val_path, rl_val)

        # Save RNG state for reproducibility
        self._save_rng_state()

        print(f"Created {len(rl_train)} RL train examples in {self.rl_train_path}")
        print(f"Created {len(rl_val)} RL val examples in {self.rl_val_path}")

        return rl_train, rl_val

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate a model on test.jsonl.

        Uses evaluation config: {model_name, batch_size, max_new_tokens, temperature, ...}

        Returns:
            Dict with evaluation results and metrics
        """
        import json
        from collections import defaultdict
        from datetime import datetime

        if not self.test_path.exists():
            raise FileNotFoundError(f"{self.test_path} not found. Run create_split() first.")

        eval_cfg = self.cfg.evaluation
        model_name = eval_cfg.model_name

        # Generate output filename
        if eval_cfg.output_name:
            output_name = eval_cfg.output_name
        else:
            # Auto-generate from model name
            model_short = model_name.split("/")[-1].replace("-", "_").lower()
            output_name = model_short

        output_path = self._artifact_path(f"eval_{output_name}.jsonl")

        print(f"Evaluating model: {model_name}")
        print(f"Test set: {self.test_path}")
        print(f"Output: {output_path}")

        # Build index for correctness checks
        self._ensure_raw_index()

        # Load test examples
        examples = load_jsonl(self.test_path)
        print(f"Loaded {len(examples)} test examples")

        # Initialize generator with eval config
        generator = VLLMGenerator(eval_cfg)

        results = []
        # Track by operand count
        stats_by_operands = defaultdict(lambda: {"correct": 0, "total": 0, "incomplete": 0, "wrong": 0, "malformed": 0, "correct_lengths": []})
        total_correct = 0
        total_incomplete = 0
        total_wrong = 0
        total_malformed = 0  # incomplete but didn't hit token limit
        correct_lengths = []
        malformed_indices = []

        for batch in generator.batch_iter(examples):
            prompts = [self._messages_to_string(ex["prompt"]) for ex in batch]
            generated = generator.generate(prompts)

            for ex, text in zip(batch, generated):
                cot, length = generator.clean_cot(text)
                is_correct = self._check_correctness_fast(ex["index"], cot)
                missing_tags = "</think>" not in cot or "<answer>" not in cot
                hit_token_limit = length >= 2047
                is_incomplete = missing_tags and hit_token_limit
                is_malformed = missing_tags and not hit_token_limit  # stopped early without hitting limit

                # Get operand count from raw index
                raw = self._raw_index.get(ex["index"], {})
                num_operands = raw.get("metadata", {}).get("num_operands", 0)

                # Update stats
                stats_by_operands[num_operands]["total"] += 1
                if is_correct:
                    total_correct += 1
                    correct_lengths.append(length)
                    stats_by_operands[num_operands]["correct"] += 1
                    stats_by_operands[num_operands]["correct_lengths"].append(length)
                elif is_incomplete:
                    total_incomplete += 1
                    stats_by_operands[num_operands]["incomplete"] += 1
                elif is_malformed:
                    total_malformed += 1
                    malformed_indices.append(ex["index"])
                    stats_by_operands[num_operands]["malformed"] += 1
                else:
                    total_wrong += 1
                    stats_by_operands[num_operands]["wrong"] += 1

                results.append({
                    "index": ex["index"],
                    "prompt": ex["prompt"],
                    "cot": cot,
                    "cot_metadata": {
                        "correct_answer": is_correct,
                        "cot_token_length": length,
                        "incomplete": is_incomplete,
                        "malformed": is_malformed,
                        "num_operands": num_operands,
                    },
                })

            # Progress update
            print(f"Processed {len(results)}/{len(examples)} - Correct: {total_correct}, Incomplete: {total_incomplete}, Malformed: {total_malformed}, Wrong: {total_wrong}")

        # Save results
        write_jsonl(output_path, results)

        # Compute final metrics
        import numpy as np

        total = len(results)
        total_complete = total - total_incomplete - total_malformed
        accuracy_raw = total_correct / total * 100 if total > 0 else 0
        accuracy_adjusted = total_correct / total_complete * 100 if total_complete > 0 else 0

        # Length stats for correct answers
        length_stats = {}
        if correct_lengths:
            arr = np.array(correct_lengths)
            length_stats = {
                "correct_length_min": int(np.min(arr)),
                "correct_length_p25": int(np.percentile(arr, 25)),
                "correct_length_median": int(np.percentile(arr, 50)),
                "correct_length_p75": int(np.percentile(arr, 75)),
                "correct_length_max": int(np.max(arr)),
                "correct_length_mean": float(np.mean(arr)),
            }

        metrics = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "total": total,
            "correct": total_correct,
            "incomplete": total_incomplete,
            "malformed": total_malformed,
            "malformed_indices": malformed_indices[:20] if malformed_indices else [],  # first 20
            "wrong": total_wrong,
            "accuracy_raw": accuracy_raw,
            "accuracy_adjusted": accuracy_adjusted,  # excludes incomplete and malformed
            **length_stats,
            "output_path": str(output_path),
        }

        # Add per-operand stats
        for n_ops in sorted(stats_by_operands.keys()):
            s = stats_by_operands[n_ops]
            complete = s["total"] - s["incomplete"] - s["malformed"]
            metrics[f"num_{n_ops}"] = s["total"]
            metrics[f"correct_{n_ops}"] = s["correct"]
            metrics[f"incomplete_{n_ops}"] = s["incomplete"]
            metrics[f"malformed_{n_ops}"] = s["malformed"]
            metrics[f"wrong_{n_ops}"] = s["wrong"]
            metrics[f"accuracy_{n_ops}_raw"] = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
            metrics[f"accuracy_{n_ops}_adjusted"] = s["correct"] / complete * 100 if complete > 0 else 0
            if s["correct_lengths"]:
                arr = np.array(s["correct_lengths"])
                metrics[f"correct_length_{n_ops}_median"] = int(np.percentile(arr, 50))
                metrics[f"correct_length_{n_ops}_mean"] = float(np.mean(arr))

        # Save metrics summary
        metrics_path = self._artifact_path(f"eval_{output_name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Print report
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"\nOverall:")
        print(f"  Total: {total}")
        print(f"  Correct: {total_correct}")
        print(f"  Incomplete (ran out of tokens): {total_incomplete}")
        print(f"  Malformed (missing tags, didn't hit limit): {total_malformed}")
        if malformed_indices:
            print(f"    Malformed indices: {malformed_indices[:10]}{'...' if len(malformed_indices) > 10 else ''}")
        print(f"  Wrong (complete but incorrect): {total_wrong}")
        print(f"  Accuracy (raw): {accuracy_raw:.1f}%")
        print(f"  Accuracy (adjusted, excl. incomplete/malformed): {accuracy_adjusted:.1f}%")

        if length_stats:
            print(f"\nCorrect answer lengths:")
            print(f"  Min: {length_stats['correct_length_min']}, 25p: {length_stats['correct_length_p25']}, "
                  f"Median: {length_stats['correct_length_median']}, 75p: {length_stats['correct_length_p75']}, "
                  f"Max: {length_stats['correct_length_max']}")

        print(f"\nBy operand count:")
        for n_ops in sorted(stats_by_operands.keys()):
            s = stats_by_operands[n_ops]
            complete = s["total"] - s["incomplete"] - s["malformed"]
            adj_acc = s["correct"] / complete * 100 if complete > 0 else 0
            median_len = int(np.median(s["correct_lengths"])) if s["correct_lengths"] else 0
            malformed_str = f", {s['malformed']} malformed" if s["malformed"] > 0 else ""
            print(f"  {n_ops} operands: {s['correct']}/{s['total']} correct, {s['incomplete']} incomplete{malformed_str}, {s['wrong']} wrong (adj. acc: {adj_acc:.1f}%, median len: {median_len})")

        print(f"\nResults: {output_path}")
        print(f"Metrics: {metrics_path}")
        print("=" * 60 + "\n")

        return metrics
