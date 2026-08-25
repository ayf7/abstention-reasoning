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
    # Training commands
    "train_sft",
    "train_rl",
    "convert_checkpoint",
]
