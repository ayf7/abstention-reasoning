"""
Pipeline commands - re-exports all commands for convenient access.

Usage:
    from pipeline import commands
    commands.create_primitives(...)
    commands.generate(...)
    commands.train_sft(...)
"""

from .data import create_primitives, create_prompts
from .inference import generate, evaluate
from .training import train_sft, train_rl, train_classifier, convert_checkpoint

__all__ = [
    # Data commands
    "create_primitives",
    "create_prompts",
    # Inference commands
    "generate",
    "evaluate",
    # Training commands
    "train_sft",
    "train_rl",
    "train_classifier",
    "convert_checkpoint",
]
