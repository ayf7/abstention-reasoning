"""
Countdown dataset manager implementing task-specific hooks.

Extends BaseManager with countdown-specific logic for:
- Puzzle generation using expression trees
- Answer extraction from <answer> tags
- Expression evaluation and comparison
- RL reward model data structure
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from data_new.core.base_manager import BaseManager, Message
from data_new.core.io_utils import write_jsonl
from data_new.countdown.generator import CountdownPuzzleGenerator

if TYPE_CHECKING:
    from omegaconf import DictConfig


class CountdownManager(BaseManager):
    """
    Manager for Countdown dataset.

    Countdown puzzles require reaching a target number using given numbers
    and basic arithmetic operations (+, -, *, /).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Initialize puzzle generator
        dataset_cfg = cfg.dataset
        self._generator = CountdownPuzzleGenerator(
            seed=cfg.task.seed,
            number_range=tuple(dataset_cfg.number_range),
            target_range=tuple(dataset_cfg.target_range),
        )

        # Load template
        self._template: Optional[str] = None

    @property
    def template(self) -> str:
        """Lazy-load prompt template."""
        if self._template is None:
            if self.template_path.exists():
                self._template = self.template_path.read_text()
            else:
                # Default template
                self._template = (
                    "Using the numbers {numbers}, create an equation that equals {target}. "
                    "You can use basic arithmetic operations (+, -, *, /) and each number "
                    "can only be used once. Show your work in <think> </think> tags. "
                    "And return the final answer in <answer> </answer> tags, "
                    "for example <answer> (1 + 2) / 3 </answer>."
                )
        return self._template

    # ========================================================================
    # Abstract method implementations
    # ========================================================================

    def create_dataset(self) -> List[Dict]:
        """
        Generate countdown puzzles and save to raw_dataset.jsonl.

        Uses dataset config: {num_samples: {4: 500, 5: 500, 6: 500}, ...}
        """
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        print("Creating countdown dataset...")

        records = []
        num_samples = self.cfg.dataset.num_samples

        for num_operands, count in num_samples.items():
            print(f"Generating {count} examples with {num_operands} operands...")
            examples = self._generator.generate_variant(int(num_operands), count)
            records.extend(examples)

        # Assign sequential indices
        for idx, rec in enumerate(records):
            rec["index"] = idx

        write_jsonl(self.raw_dataset_path, records)

        # Save RNG state for reproducibility
        self._save_rng_state()

        print(f"Created {len(records)} countdown examples in {self.raw_dataset_path}")

        return records

    def _build_prompt_messages(self, raw_record: Dict) -> List[Message]:
        """
        Build messages list for countdown prompt.

        Returns:
            [system, user, assistant_prefix] messages
        """
        target = raw_record["question"]["target"]
        numbers = raw_record["question"]["numbers"]

        user_content = self.template.format(
            target=target,
            numbers=numbers,
        )

        return [
            {"role": "system", "content": self.cfg.prompt.system_message},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": self.cfg.prompt.assistant_prefix},
        ]

    def _extract_answer(self, cot_text: str) -> Optional[str]:
        """Extract expression from <answer> tags."""
        match = re.search(
            r"<answer>(.*?)</answer>",
            cot_text,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # Fallback: look for "Answer:" pattern
        match = re.search(r"Answer:\s*(.+?)(?:\n|$)", cot_text)
        if match:
            return match.group(1).strip()

        return None

    def _compare_answers(self, predicted: Optional[str], raw_record: Dict) -> bool:
        """Evaluate expression and compare with target."""
        if not predicted:
            return False

        target = raw_record["question"]["target"]
        numbers = raw_record["question"]["numbers"]

        try:
            result = self._evaluate_expression(predicted, numbers)
            return result == target
        except Exception:
            return False

    def _get_reward_model_data(self, raw_record: Dict) -> Dict[str, Any]:
        """Build reward model data for RL training."""
        return {
            "ground_truth": {
                "target": raw_record["question"]["target"],
                "numbers": raw_record["question"]["numbers"],
                "solution_expr": raw_record["answer"],
                "hint_exprs": raw_record["metadata"].get("hint_exprs", []),
            }
        }

    # ========================================================================
    # Countdown-specific helpers
    # ========================================================================

    def _evaluate_expression(
        self,
        expr: str,
        available_numbers: List[int],
    ) -> Optional[int]:
        """
        Safely evaluate an expression using only available numbers.

        Args:
            expr: Arithmetic expression string
            available_numbers: Numbers that can be used

        Returns:
            Result if valid, None otherwise
        """
        try:
            expr = expr.strip()

            # Extract all numbers from the expression
            numbers_in_expr = [int(n) for n in re.findall(r"\b\d+\b", expr)]

            # Check that all numbers used are from the available list
            available_copy = available_numbers.copy()
            for num in numbers_in_expr:
                if num in available_copy:
                    available_copy.remove(num)
                else:
                    return None  # Number not available or used twice

            # Safe eval with no builtins
            result = eval(expr, {"__builtins__": {}}, {})

            # Check if result is an integer
            if isinstance(result, (int, float)) and result == int(result):
                return int(result)

            return None
        except Exception:
            return None
