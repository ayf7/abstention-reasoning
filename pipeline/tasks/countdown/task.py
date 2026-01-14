"""Countdown task implementation."""

import re

from pipeline.tasks.base import BaseTask
from pipeline.core.utils import safe_eval


class CountdownTask(BaseTask):
    """
    Countdown numbers game task.

    Goal: Use given numbers with +, -, *, / to reach a target value.
    Each number can only be used once.
    """

    name = "countdown"

    # Default system message
    system_message = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then provides the user with the answer."
    )

    def create_primitives(self, num_puzzles: int, seed: int = 42) -> list[dict]:
        """Generate countdown puzzles."""
        from .generator import generate_puzzles

        return generate_puzzles(
            num_puzzles=num_puzzles,
            seed=seed,
            operand_distribution={4: 0.33, 5: 0.33, 6: 0.34},
        )

    def format_prompt(
        self,
        primitive: dict,
        template: str,
        include_assistant_prefix: bool = True,
    ) -> list[dict]:
        """Format countdown puzzle into chat messages."""
        # Substitute variables in template
        content = template.replace("{{target}}", str(primitive["target"]))
        content = template.replace("{target}", str(primitive["target"]))
        content = content.replace("{{numbers}}", str(primitive["numbers"]))
        content = content.replace("{numbers}", str(primitive["numbers"]))

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": content},
        ]

        if include_assistant_prefix:
            messages.append({
                "role": "assistant",
                "content": "<think> Let me solve this step by step.",
            })

        return messages

    def check_correctness(
        self,
        primitive: dict,
        generation: str,
    ) -> tuple[bool, dict]:
        """
        Check if the generated expression correctly solves the puzzle.

        Validates:
        1. Answer can be parsed from <answer> tags
        2. Expression only uses available numbers
        3. Each number used at most once
        4. Expression evaluates to target
        """
        answer = self.extract_answer(generation)

        if answer is None:
            return False, {
                "predicted_answer": None,
                "error": "no_answer_tag",
            }

        try:
            # Extract numbers used in expression
            numbers_used = [int(n) for n in re.findall(r'\b\d+\b', answer)]
            available = primitive["numbers"].copy()

            # Check each number is available and used at most once
            for num in numbers_used:
                if num not in available:
                    return False, {
                        "predicted_answer": answer,
                        "error": "invalid_number",
                        "invalid_number": num,
                    }
                available.remove(num)

            # Evaluate expression
            result = safe_eval(answer)

            # Check if result matches target
            is_correct = (result == primitive["target"])

            return is_correct, {
                "predicted_answer": answer,
                "result": result,
                "target": primitive["target"],
            }

        except Exception as e:
            return False, {
                "predicted_answer": answer,
                "error": str(e),
            }

    def compute_reward(
        self,
        primitive: dict,
        generation: str,
    ) -> tuple[float, dict]:
        """
        Compute reward for RL training.

        Currently binary: 1.0 if correct, 0.0 otherwise.
        Could be extended for partial credit (e.g., close to target).
        """
        is_correct, meta = self.check_correctness(primitive, generation)
        reward = 1.0 if is_correct else 0.0
        return reward, {"correct": is_correct, **meta}
