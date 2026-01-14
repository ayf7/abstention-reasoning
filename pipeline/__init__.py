"""
Pipeline module for data generation, training, and evaluation.

Usage:
    python -m pipeline task=countdown create_primitives
    python -m pipeline task=countdown create_prompts --split=sft
    python -m pipeline task=countdown generate --prompts=prompts/sft.json --model=Qwen/Qwen3-14B
    python -m pipeline task=countdown train_sft --dataset=datasets/sft_qwen3-14b.json
    python -m pipeline task=countdown evaluate --model=models/sft
"""

__version__ = "0.1.0"
