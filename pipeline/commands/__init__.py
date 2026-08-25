"""
Pipeline commands - re-exports all commands for convenient access.

Usage:
    from pipeline import commands
    commands.create_primitives(...)
    commands.generate(...)
    commands.train_sft(...)
"""

from .data import create_primitives, create_prompts, create_verify_prompts, create_ood_prompts, OOD_DATASETS
from .inference import generate, evaluate, analyze
from .analysis import analyze_abstention
from .judge import preprocess_for_verify_judge, judge_verify, judge_correctness
from .comparison import compare_sampling
from .training import train_sft, train_rl, convert_checkpoint

__all__ = [
    # Data commands
    "create_primitives",
    "create_prompts",
    "create_verify_prompts",
    "create_ood_prompts",
    "OOD_DATASETS",
    # Inference commands
    "generate",
    "evaluate",
    "analyze",
    "analyze_abstention",
    "preprocess_for_verify_judge",
    "judge_verify",
    "judge_correctness",
    "compare_sampling",
    # Training commands
    "train_sft",
    "train_rl",
    "convert_checkpoint",
]
