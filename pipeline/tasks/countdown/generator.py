"""
Countdown puzzle generator using expression trees.

Generates countdown-style puzzles where the goal is to reach a target number
using a set of numbers and basic arithmetic operations.
"""

import operator
import random
from typing import Optional


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
        value: int | str,
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


def generate_puzzles(
    num_puzzles: int,
    seed: int = 42,
    operand_distribution: dict[int, float] | None = None,
    number_range: tuple[int, int] = (1, 100),
    target_range: tuple[int, int] = (10, 1000),
) -> list[dict]:
    """
    Generate countdown puzzles.

    Args:
        num_puzzles: Total number of puzzles to generate
        seed: Random seed
        operand_distribution: Dict mapping num_operands -> proportion (e.g., {4: 0.33, 5: 0.33, 6: 0.34})
        number_range: Range for individual numbers
        target_range: Range for target values

    Returns:
        List of puzzle dicts with 'index', 'target', 'numbers', 'answer', 'variant'
    """
    if operand_distribution is None:
        operand_distribution = {4: 0.33, 5: 0.33, 6: 0.34}

    rng = random.Random(seed)
    generator = _PuzzleGenerator(rng, number_range, target_range)

    # Calculate count per variant
    counts = {}
    remaining = num_puzzles
    variants = list(operand_distribution.keys())
    for i, (num_ops, prop) in enumerate(operand_distribution.items()):
        if i == len(operand_distribution) - 1:
            counts[num_ops] = remaining
        else:
            count = int(num_puzzles * prop)
            counts[num_ops] = count
            remaining -= count

    # Generate puzzles for each variant
    all_puzzles = []
    for num_ops, count in counts.items():
        puzzles = generator.generate_variant(num_ops, count)
        all_puzzles.extend(puzzles)

    # Shuffle and assign indices
    rng.shuffle(all_puzzles)
    for i, puzzle in enumerate(all_puzzles):
        puzzle["index"] = i

    return all_puzzles


class _PuzzleGenerator:
    """Internal puzzle generator."""

    def __init__(
        self,
        rng: random.Random,
        number_range: tuple[int, int],
        target_range: tuple[int, int],
    ):
        self.rng = rng
        self.number_range = number_range
        self.target_range = target_range

    def generate_variant(self, num_operands: int, count: int) -> list[dict]:
        """Generate puzzles for a specific number of operands."""
        records = []
        attempts = 0
        max_attempts = count * 100

        while len(records) < count and attempts < max_attempts:
            attempts += 1

            target = self.rng.randint(*self.target_range)
            tree = self._build_random_expr_tree(num_operands, target)
            if tree is None:
                continue

            result = self._evaluate_expr_tree(tree)
            if result != target:
                continue

            numbers = self._extract_leaf_numbers(tree)

            # Ensure all numbers are unique and within range
            if len(set(numbers)) != num_operands:
                continue
            if any(n < self.number_range[0] or n > self.number_range[1] for n in numbers):
                continue
            if target in numbers:
                continue

            record = {
                "index": 0,  # Set later
                "target": target,
                "numbers": numbers,
                "answer": str(tree),
                "variant": f"{num_operands}_operands",
            }
            records.append(record)

        if len(records) < count:
            print(f"Warning: generated {len(records)}/{count} for {num_operands} operands")

        return records

    def _build_random_expr_tree(self, num_numbers: int, target: int) -> Optional[ExprNode]:
        """Build a random expression tree that evaluates to target."""
        if num_numbers == 1:
            return ExprNode(target)

        split = self.rng.randint(1, num_numbers - 1)
        left_nums = split
        right_nums = num_numbers - split

        ops = OP_SYMBOLS.copy()
        self.rng.shuffle(ops)

        for op in ops:
            left_target, right_target = self._find_operands_for_target(target, op)
            if left_target is None:
                continue

            left_tree = self._build_random_expr_tree(left_nums, left_target)
            right_tree = self._build_random_expr_tree(right_nums, right_target)

            if left_tree is None or right_tree is None:
                continue

            return ExprNode(op, left_tree, right_tree)

        return None

    def _find_operands_for_target(
        self, target: int, op_symbol: str, max_attempts: int = 1000
    ) -> tuple[Optional[int], Optional[int]]:
        """Find a, b such that a <op> b == target."""
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

            if lo <= a <= hi and lo <= b <= hi:
                result = op_func(a, b)
                if result == target:
                    return a, b

        return None, None

    @staticmethod
    def _extract_leaf_numbers(node: ExprNode) -> list[int]:
        """Extract all leaf numbers from the expression tree."""
        if node.is_leaf():
            return [node.value]
        return (
            _PuzzleGenerator._extract_leaf_numbers(node.left)
            + _PuzzleGenerator._extract_leaf_numbers(node.right)
        )

    @staticmethod
    def _evaluate_expr_tree(node: ExprNode) -> Optional[int]:
        """Evaluate the expression tree."""
        if node.is_leaf():
            return node.value
        left_val = _PuzzleGenerator._evaluate_expr_tree(node.left)
        right_val = _PuzzleGenerator._evaluate_expr_tree(node.right)
        if left_val is None or right_val is None:
            return None
        return OPS[node.value](left_val, right_val)
