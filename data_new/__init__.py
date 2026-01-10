"""
data_new: Clean data pipeline with Hydra configuration.

Usage:
    python -m data_new command=create_dataset
    python -m data_new command=create_split
    python -m data_new command=create_generations
    python -m data_new command=run_sft
    python -m data_new command=create_rl_data

See config/base.yaml and config/countdown.yaml for configuration options.
"""
from data_new.core import BaseManager, VLLMGenerator

__all__ = ["BaseManager", "VLLMGenerator"]
