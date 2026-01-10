"""
Countdown puzzle generator using expression trees.

Generates countdown-style puzzles where the goal is to reach a target number
using a set of numbers and basic arithmetic operations.
"""
from __future__ import annotations

import operator
import random
from typing import List, Optional, Tuple, Union


# Operator definitions
OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": lambda a, b: a // b if b != 0 and a % b == 0 else None,
}
OP_SYMBOLS = list(OPS.keys())


class ExprNode:
    """Expression tree node for countdown expressions."""

    def __init__(
        self,
        value: Union[int, str],
        left: Optional["ExprNode"] = None,
        right: Optional["ExprNode"] = None,
    ):
        self.value = value  # int (leaf) or operator string
        self.left = left
        self.right = right

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def __str__(self) -> str:
        if self.is_leaf():
            return str(self.value)
        return f"({self.left} {self.value} {self.right})"


class CountdownPuzzleGenerator:
    """
    Generator for countdown puzzles using expression trees.

    Creates puzzles where:
    - A target number must be reached
    - A set of numbers is provided
    - Only +, -, *, / operations are allowed
    - Each number can only be used once
    """

    def __init__(
        self,
        seed: int = 42,
        number_range: Tuple[int, int] = (1, 100),
        target_range: Tuple[int, int] = (10, 1000),
    ):
        self.rng = random.Random(seed)
        self.number_range = number_range
        self.target_range = target_range

    def generate_variant(self, num_operands: int, count: int) -> List[dict]:
        """
        Generate countdown examples for a specific number of operands.

        Args:
            num_operands: Number of numbers to use in each puzzle
            count: Number of puzzles to generate

        Returns:
            List of puzzle records
        """
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

            record = {
                "index": 0,  # Will be set by manager
                "question": {
                    "target": target,
                    "numbers": numbers,
                },
                "answer": solution_expr,
                "metadata": {
                    "variant": f"{num_operands}_operands",
                    "num_operands": num_operands,
                    "hint_exprs": hints,
                },
            }

            records.append(record)

        if len(records) < count:
            print(f"Warning: generated {len(records)}/{count} for {num_operands} operands")

        return records

    def _build_random_expr_tree(
        self,
        num_numbers: int,
        target: int,
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
        max_attempts: int = 1000,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Find two numbers a and b such that a <op> b == target."""
        op_func = OPS[op_symbol]
        lo, hi = self.number_range

        for _ in range(max_attempts):
            if op_symbol == "+":
                a = self.rng.randint(lo, hi)
                b = target - a
            elif op_symbol == "-":
                a = self.rng.randint(lo, hi)
                b = a - target
            elif op_symbol == "*":
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
            elif op_symbol == "/":
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
        return (
            CountdownPuzzleGenerator._extract_leaf_numbers(node.left)
            + CountdownPuzzleGenerator._extract_leaf_numbers(node.right)
        )

    @staticmethod
    def _evaluate_expr_tree(node: ExprNode) -> Optional[int]:
        """Evaluate the expression tree."""
        if node.is_leaf():
            return node.value
        left_val = CountdownPuzzleGenerator._evaluate_expr_tree(node.left)
        right_val = CountdownPuzzleGenerator._evaluate_expr_tree(node.right)
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
        hints += CountdownPuzzleGenerator._extract_hints(node.left)
        hints += CountdownPuzzleGenerator._extract_hints(node.right)
        hints.append(str(node))
        return hints
