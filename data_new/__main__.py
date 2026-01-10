"""
Hydra CLI entrypoint for data_new module.

Usage:
    python -m data_new task=countdown command=create_dataset
    python -m data_new task=countdown command=create_split
    python -m data_new task=countdown command=create_generations
    python -m data_new task=countdown command=analyze_generations
    python -m data_new task=countdown command=retry_failed_generations
    python -m data_new task=countdown command=run_sft
    python -m data_new task=countdown command=create_rl_data
    python -m data_new task=countdown command=run_all
"""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Task registry: maps task name to manager class path
MANAGERS = {
    "countdown": "data_new.countdown.manager.CountdownManager",
    # Future tasks:
    # "connections": "data_new.connections.manager.ConnectionsManager",
    # "knights_and_knaves": "data_new.knights_and_knaves.manager.KnightsAndKnavesManager",
}


def get_manager(cfg: DictConfig):
    """Dynamically load and instantiate the manager for the configured task."""
    from importlib import import_module

    task = cfg.task.name
    if task not in MANAGERS:
        available = list(MANAGERS.keys())
        raise ValueError(f"Unknown task: {task}. Available: {available}")

    module_path, class_name = MANAGERS[task].rsplit(".", 1)
    module = import_module(module_path)
    cls = getattr(module, class_name)
    return cls(cfg)


def resolve_paths(cfg: DictConfig) -> None:
    """Auto-resolve paths if not explicitly set."""
    if cfg.paths.root_dir is None:
        # Set root_dir to repo root (parent of data_new/)
        root = Path(__file__).parent.parent
        cfg.paths.root_dir = str(root)


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for data_new CLI.

    Commands:
        create_dataset: Generate raw_dataset.jsonl
        create_split: Create train.jsonl and test.jsonl
        create_generations: Generate CoTs for train.jsonl
        analyze_generations: Print statistics on CoT lengths and accuracy
        retry_failed_generations: Retry failed CoT generations
        run_sft: Run SFT training on correct examples
        create_rl_data: Create rl_train.parquet and rl_val.parquet
        run_all: Run create_dataset and create_split
    """
    # Resolve paths
    resolve_paths(cfg)
    OmegaConf.resolve(cfg)

    # Print config summary
    print("=" * 60)
    print(f"Task: {cfg.task.name}")
    print(f"Command: {cfg.command}")
    print(f"Artifacts: {cfg.paths.artifacts_dir}")
    print("=" * 60)

    # Get manager
    manager = get_manager(cfg)

    # Execute command
    command = cfg.command

    if command == "create_dataset":
        manager.create_dataset()

    elif command == "create_split":
        manager.create_split()

    elif command == "create_generations":
        manager.create_generations()

    elif command == "retry_failed_generations":
        manager.retry_failed_generations()

    elif command == "run_sft":
        manager.run_sft()

    elif command == "create_rl_data":
        manager.create_rl_data()

    elif command == "analyze_generations":
        manager.analyze_generations()

    elif command == "reevaluate_correctness":
        manager.reevaluate_correctness()

    elif command == "evaluate":
        manager.evaluate()

    elif command == "run_all":
        print("Running dataset creation and split...")
        manager.create_dataset()
        manager.create_split()
        print("\nDataset and splits created.")
        print("\nTo continue the pipeline, run:")
        print("  python -m data_new command=create_generations")
        print("  python -m data_new command=run_sft")
        print("  python -m data_new command=create_rl_data")

    else:
        available = [
            "create_dataset",
            "create_split",
            "create_generations",
            "analyze_generations",
            "reevaluate_correctness",
            "retry_failed_generations",
            "run_sft",
            "evaluate",
            "create_rl_data",
            "run_all",
        ]
        raise ValueError(f"Unknown command: {command}. Available: {available}")


if __name__ == "__main__":
    main()
