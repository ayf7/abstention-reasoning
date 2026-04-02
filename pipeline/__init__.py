"""
Pipeline module for data generation, training, and evaluation.

Usage:
    python -m pipeline create_primitives --task countdown --output primitives.json
    python -m pipeline create_prompts --task countdown --primitives primitives.json --output prompts/
    python -m pipeline generate --task countdown --prompts prompts/sft.json --output datasets/sft.json --model Qwen/Qwen3-14B
    python -m pipeline train_sft --task countdown --dataset datasets/sft.json --output models/sft --base-model Qwen/Qwen2.5-1.5B
    python -m pipeline evaluate --task countdown --prompts prompts/eval.json --output results/eval.json --model models/sft
"""

__version__ = "0.1.0"
