"""
Standardized manager for NYT Connections dataset.

Manages the full pipeline from CSV -> raw_dataset -> train/test splits ->
CoT generation -> SFT training -> RL data creation.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

from data.dataset_manager import DatasetManager
from data.prompt_loader import generate_connections_prompt


class ConnectionsManager(DatasetManager):
    """
    Manages the Connections dataset following the standardized framework.

    Directory structure:
        data/connections/
            connections_data.csv     # Raw CSV data (expected to exist)
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
    ):
        if path is None:
            path = Path(__file__).resolve().parent
        if prompt_template is None:
            prompt_template = path / "generation_template.txt"

        super().__init__(path, prompt_template)

        self.csv_path = path / "connections_data.csv"
        self.test_split_ratio = test_split_ratio
        self.seed = seed
        self.rng = random.Random(seed)

    # ========================================================================
    # Dataset Creation
    # ========================================================================

    def create_dataset(self):
        """
        Creates raw_dataset.jsonl from connections_data.csv.

        Each record in raw_dataset.jsonl:
        {
            "index": <int>,
            "question": <dict with words, answers, etc.>,
            "answer": <str>,  # Canonical answer format
            "metadata": {
                "game_id": <int>,
                "puzzle_date": <str>,
                "contest": <str>,
                "total_words": <int>,
                "group_count": <int>,
                "words_per_group": <int>
            }
        }
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.artifact_dir / "raw_dataset.jsonl"

        print(f"Creating dataset from {self.csv_path}...")

        # Load CSV and group by puzzle
        puzzles = self._load_puzzles_from_csv()

        # Sort chronologically
        sorted_puzzles = self._sort_puzzles(puzzles)

        # Build records
        records = []
        for idx, (game_id, puzzle_date, rows) in enumerate(sorted_puzzles):
            record = self._build_puzzle_record(idx, game_id, puzzle_date, rows)
            records.append(record)

        # Write to JSONL
        with output_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"Created {len(records)} puzzles in {output_path}")
        return records

    def _load_puzzles_from_csv(self) -> Dict[int, List[dict]]:
        """Load CSV and group rows by Game ID."""
        puzzles = defaultdict(list)
        with self.csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                game_id = int(row["Game ID"])
                puzzles[game_id].append(row)
        return puzzles

    def _sort_puzzles(self, puzzles: Dict[int, List[dict]]) -> List[Tuple[int, str, List[dict]]]:
        """Sort puzzles chronologically."""
        sorted_puzzles = []
        for game_id, rows in puzzles.items():
            if not rows:
                continue
            puzzle_date = rows[0]["Puzzle Date"]
            date_ms = self._parse_date_ms(puzzle_date)
            sorted_puzzles.append((date_ms, game_id, puzzle_date, rows))

        sorted_puzzles.sort(key=lambda t: (t[0], t[1]))
        return [(gid, pdate, rows) for _, gid, pdate, rows in sorted_puzzles]

    def _build_puzzle_record(
        self,
        idx: int,
        game_id: int,
        puzzle_date: str,
        rows: List[dict]
    ) -> dict:
        """Build a single puzzle record from CSV rows."""
        # Group words by group name
        groups = defaultdict(list)
        for r in rows:
            group_name = self._normalize_text(r["Group Name"])
            word = self._normalize_text(r["Word"])
            row_idx = int(r["Starting Row"])
            col_idx = int(r["Starting Column"])
            groups[group_name].append((row_idx, col_idx, word))

        # Sort groups alphabetically; words by board position
        answers = []
        all_words = []
        for group_name in sorted(groups):
            word_entries = sorted(groups[group_name], key=lambda t: (t[0], t[1]))
            words = [w for _, _, w in word_entries]
            all_words.extend(words)
            answers.append({
                "answerDescription": group_name,
                "words": words
            })

        # Shuffle words for the puzzle
        shuffled_words = list(all_words)
        self.rng.shuffle(shuffled_words)

        # Build canonical answer format
        canonical_answer = self._format_canonical_answer(answers)

        return {
            "index": idx,
            "question": {
                "words": shuffled_words,
                "answers": answers,
            },
            "answer": canonical_answer,
            "metadata": {
                "game_id": game_id,
                "puzzle_date": puzzle_date,
                "contest": f"NYT Connections {game_id} - {puzzle_date}",
                "total_words": len(all_words),
                "group_count": len(answers),
                "words_per_group": len(answers[0]["words"]) if answers else 4,
            }
        }

    def _format_canonical_answer(self, answers: List[dict]) -> str:
        """Format the canonical answer string."""
        groups = []
        for ans in answers:
            words_upper = [w.upper() for w in ans["words"]]
            group_str = "[" + ", ".join(words_upper) + "]"
            groups.append(group_str)
        return "{" + ", ".join(groups) + "}"

    @staticmethod
    def _parse_date_ms(date_str: str) -> int:
        """Parse date string to milliseconds."""
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text (remove curly quotes, etc.)."""
        if text is None:
            return ""
        text = unicodedata.normalize("NFC", text)
        replacements = {
            "\u201c": '"', "\u201d": '"',
            "\u2018": "'", "\u2019": "'",
            "\u2013": "-", "\u2014": "-",
            "\u00a0": " ",
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return text.strip()

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
        print(f"Model: {model_name}, Batch size: {batch_size}")

        # Load model
        tokenizer, model = self._init_model(model_name)

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

        updated_examples = []
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
                tokenizer, model, prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )

            # Process generations
            for j, (ex, gen_text) in enumerate(zip(batch, generated_texts)):
                cot, cot_length = self._clean_cot(gen_text, tokenizer)

                # Check correctness (load raw dataset to get answer)
                is_correct = self._check_correctness(ex["index"], cot)

                ex["cot"] = cot
                ex["cot_metadata"] = {
                    "correct_answer": is_correct,
                    "cot_token_length": cot_length,
                }

                updated_examples.append(ex)
                processed_indices.add(ex["index"])

        # Write back
        if in_place:
            # Merge with existing examples
            all_examples = self._merge_examples(examples, updated_examples)
            self._write_jsonl(train_path, all_examples)
            print(f"Updated {train_path} with {len(updated_examples)} new generations")
        elif output_file:
            self._write_jsonl(output_file, updated_examples)
            print(f"Wrote {len(updated_examples)} generations to {output_file}")

        return updated_examples

    def _init_model(self, model_name: str) -> Tuple[Any, Any]:
        """Initialize model and tokenizer."""
        print(f"Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        return tokenizer, model

    def _generate_batch(
        self,
        tokenizer,
        model,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[str]:
        """Generate text for a batch of prompts."""
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

        # Decode only the generated tokens
        generated_texts = []
        input_length = inputs["input_ids"].shape[-1]
        for output in output_ids:
            generated_ids = output[input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(generated_text)

        return generated_texts

    def _clean_cot(self, text: str, tokenizer) -> Tuple[str, int]:
        """Clean generated text and compute token length."""
        # Remove end markers
        if "<|endoftext|>" in text:
            text = text.split("<|endoftext|>", 1)[0]

        # Keep only up to the answer block
        answer_match = re.search(r"<answer>.*?</answer>", text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            text = text[:answer_match.end()]

        text = text.strip()
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
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
        def preprocess(example):
            return {
                "prompt": example["prompt"],
                "completion": example["cot"]
            }

        processed = filtered.map(
            preprocess,
            remove_columns=["cot", "cot_metadata"]
        )

        # Split for validation
        split_dataset = processed.train_test_split(test_size=0.1, seed=self.seed)

        # Setup training
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
            messages = [
                {"role": "user", "content": ex["prompt"]}
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
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B", help="Model name for generation")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--num-samples", type=int, help="Number of samples for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Initialize manager
    manager = ConnectionsManager(path=args.path, seed=args.seed)

    if args.command == "create_dataset":
        manager.create_dataset()

    elif args.command == "create_split":
        manager.create_split()

    elif args.command == "create_generations":
        manager.create_generations(
            model_name=args.model,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
        )

    elif args.command == "run_sft":
        manager.run_sft(model_name=args.model)

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
