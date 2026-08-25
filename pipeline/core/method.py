"""Method configuration - bundles template variant + reward function + artifact paths."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .utils import model_short_name


# Repo root, derived from this file's location: <repo>/pipeline/core/method.py.
# Config, template and artifact lookups used to be relative to the process's
# working directory, so every command silently required being run from the repo
# root and failed with "method not found" (or created a stray empty artifacts/)
# anywhere else.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Default artifacts root
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"

# Roots for repo-owned lookups (method configs and task templates)
CONFIGS_ROOT = REPO_ROOT / "pipeline" / "configs" / "methods"
TASKS_ROOT = REPO_ROOT / "pipeline" / "tasks"

# External storage for models (symlinked from artifacts)
# Model weights used to live outside the repo at /share/goyal/ayf7/models and
# were symlinked in per run. The whole artifacts tree now lives on shared
# storage (artifacts -> /share/goyal/ayf7/artifacts), so run directories are
# plain directories inside it and the indirection is no longer needed.
# Set EXTERNAL_MODELS_ROOT env var to re-enable the symlink behavior.
EXTERNAL_MODELS_ROOT = (
    Path(os.environ["EXTERNAL_MODELS_ROOT"])
    if "EXTERNAL_MODELS_ROOT" in os.environ
    else None
)


@dataclass
class Method:
    """
    Configuration for a pipeline method.

    A method defines a consistent configuration across the full pipeline:
    - template_variant: Which template variant to use (e.g., "simple", "simple_abstention")
    - reward_function: Name of reward function for RL training
    - reward_kwargs: Additional arguments for reward function
    - multi_turn: Enable multi-turn hint generation in RL (default: False)
    - assistant_prefix: Prefix for assistant responses (used in SFT and RL)
    - mask_response_tokens: Mask <response>...</response> tokens during SFT (default: False)

    Methods also provide auto-derived artifact paths based on task and method name.
    """

    name: str
    template_variant: str
    reward_function: str = "compute_score"
    reward_kwargs: dict[str, Any] = field(default_factory=dict)
    multi_turn: bool = False  # Enable multi-turn hint generation
    max_turns: int = 6  # RL-only: verl's rollout turn cap. generate/evaluate
                        # are bounded by max_new_tokens, not by a turn count.
    assistant_prefix: str | None = None  # If None, use task's default
    mask_response_tokens: bool = False  # Mask <response>...</response> in SFT
    interaction_name: str | None = None  # Interaction name override (default: {task}_{method})
    max_hints: int | None = None  # Maximum number of hints to give during RL rollout (None = unlimited)
    reward_manager: str = "naive"  # Reward manager type: "naive" or "batch"
    stop_strings: list[str] | None = None  # Custom stop strings for generation (None = default)

    # Backwards compatibility alias
    @property
    def allow_hint(self) -> bool:
        """Alias for multi_turn (backwards compatibility)."""
        return self.multi_turn

    @classmethod
    def load(cls, name_or_path: str, task_name: str) -> "Method":
        """
        Load a method config.

        Args:
            name_or_path: Either a method name (e.g., "simple", "simple_abstention")
                          or a path to a YAML config file
            task_name: Task name (used to find config in standard location)

        Returns:
            Method instance

        Lookup order:
            1. If name_or_path is a file path, load directly
            2. Otherwise, look in pipeline/configs/methods/{task}/{name}.yaml
        """
        path = Path(name_or_path)

        # Check if it's a direct path
        if path.exists() and path.is_file():
            config_path = path
        else:
            # Look in standard location
            config_path = CONFIGS_ROOT / task_name / f"{name_or_path}.yaml"
            if not config_path.exists():
                available = cls.list_methods(task_name)
                raise FileNotFoundError(
                    f"Method '{name_or_path}' not found for task '{task_name}'. "
                    f"Available methods: {available}"
                )

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return cls(
            name=data.get("name", name_or_path),
            template_variant=data["template_variant"],
            reward_function=data.get("reward_function", "compute_score"),
            reward_kwargs=data.get("reward_kwargs", {}),
            multi_turn=data.get("multi_turn", False),
            max_turns=data.get("max_turns", 6),
            assistant_prefix=data.get("assistant_prefix"),
            mask_response_tokens=data.get("mask_response_tokens", False),
            interaction_name=data.get("interaction_name"),
            max_hints=data.get("max_hints"),
            reward_manager=data.get("reward_manager", "naive"),
            stop_strings=data.get("stop_strings"),
        )

    @staticmethod
    def list_methods(task_name: str) -> list[str]:
        """List available methods for a task."""
        methods_dir = CONFIGS_ROOT / task_name
        if not methods_dir.exists():
            return []
        return [p.stem for p in methods_dir.glob("*.yaml")]

    def get_template_path(self, task_name: str, split: str) -> Path:
        """Get the template path for a given split."""
        return TASKS_ROOT / task_name / "templates" / self.template_variant / f"{split}.txt"

    def load_template(self, task_name: str, split: str) -> str:
        """Load the template content for a given split."""
        template_path = self.get_template_path(task_name, split)
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        with open(template_path) as f:
            return f.read()

    # =========================================================================
    # Artifact path utilities
    # =========================================================================

    def artifacts_dir(self, task_name: str) -> Path:
        """Get the artifacts directory for this method."""
        return ARTIFACTS_ROOT / task_name / self.name

    def prompts_dir(self, task_name: str) -> Path:
        """Get the prompts directory."""
        return self.artifacts_dir(task_name) / "prompts"

    def prompts_path(self, task_name: str, split: str) -> Path:
        """Get the prompts path for a specific split."""
        ext = ".parquet" if split.startswith("rl") else ".json"
        return self.prompts_dir(task_name) / f"{split}{ext}"

    def datasets_dir(self, task_name: str) -> Path:
        """Get the datasets directory."""
        return self.artifacts_dir(task_name) / "datasets"

    def dataset_path(self, task_name: str, split: str, model_name: str) -> Path:
        """Get the dataset path for a specific split and model."""
        model_slug = model_short_name(model_name)
        return self.datasets_dir(task_name) / f"{split}_{model_slug}.json"

    def models_dir(self, task_name: str) -> Path:
        """Get the models directory."""
        return self.artifacts_dir(task_name) / "models"

    def _ensure_run_dir(self, run_dir: Path, task_name: str) -> Path:
        """
        Ensure a run directory exists, using external storage if configured.

        If EXTERNAL_MODELS_ROOT is set and the run directory doesn't already
        exist, creates the target directory on external storage and symlinks
        the run dir to it. This operates at the run level (e.g.,
        models/sft/{run_id}/) so that new runs go to external storage even
        if older runs exist locally.

        Args:
            run_dir: The run directory path (e.g., models/sft/{run_id}/)
            task_name: Task name

        Returns:
            Path to the run directory.
        """
        # Already exists (real dir or symlink) -> leave as-is
        if run_dir.exists() or run_dir.is_symlink():
            return run_dir

        # External storage configured -> create symlink
        if EXTERNAL_MODELS_ROOT is not None:
            # Mirror the relative path under external storage
            # run_dir is like: artifacts/{task}/{method}/models/sft/{run_id}
            # external is:     /share/goyal/ayf7/models/{task}/{method}/sft/{run_id}
            rel_to_models = run_dir.relative_to(self.models_dir(task_name))
            external_path = EXTERNAL_MODELS_ROOT / task_name / self.name / rel_to_models
            external_path.mkdir(parents=True, exist_ok=True)
            # Ensure parent dir exists locally
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            run_dir.symlink_to(external_path)
            print(f"Created symlink: {run_dir} -> {external_path}")
        else:
            run_dir.mkdir(parents=True, exist_ok=True)

        return run_dir

    def ensure_sft_run_dir(self, task_name: str, run_id: str | None = None) -> Path:
        """Ensure the SFT run directory exists, symlinking to external storage if needed."""
        return self._ensure_run_dir(self.sft_run_dir(task_name, run_id), task_name)

    def ensure_rl_run_dir(self, task_name: str, run_id: str | None = None) -> Path:
        """Ensure the RL run directory exists, symlinking to external storage if needed."""
        return self._ensure_run_dir(self.rl_run_dir(task_name, run_id), task_name)

    def sft_run_dir(self, task_name: str, run_id: str | None = None) -> Path:
        """
        Get the SFT run directory.

        Args:
            task_name: Name of task
            run_id: Run identifier (default: "default")

        Returns:
            Path to models/sft/{run_id}/
        """
        run_id = run_id or "default"
        return self.models_dir(task_name) / "sft" / run_id

    def sft_model_path(self, task_name: str, run_id: str | None = None) -> Path:
        """
        Get the SFT model path.

        Args:
            task_name: Name of task
            run_id: Run identifier (default: "default")

        Returns:
            Path to models/sft/{run_id}/model/
        """
        return self.sft_run_dir(task_name, run_id) / "model"

    def rl_run_dir(self, task_name: str, run_id: str | None = None) -> Path:
        """
        Get the RL run directory.

        Args:
            task_name: Name of task
            run_id: Run identifier (default: "default")

        Returns:
            Path to models/rl/{run_id}/
        """
        run_id = run_id or "default"
        return self.models_dir(task_name) / "rl" / run_id

    def rl_checkpoints_dir(self, task_name: str, run_id: str | None = None) -> Path:
        """Get the RL checkpoints directory: models/rl/{run_id}/checkpoints/"""
        return self.rl_run_dir(task_name, run_id) / "checkpoints"

    def rl_rollouts_dir(self, task_name: str, run_id: str | None = None) -> Path:
        """Get the RL rollouts directory: models/rl/{run_id}/rollouts/"""
        return self.rl_run_dir(task_name, run_id) / "rollouts"

    def rl_model_path(self, task_name: str, run_id: str | None = None) -> Path:
        """
        Get the RL model path.

        Args:
            task_name: Name of task
            run_id: Run identifier (default: "default")

        Returns:
            Path to models/rl/{run_id}/model/
        """
        return self.rl_run_dir(task_name, run_id) / "model"
    def results_dir(self, task_name: str) -> Path:
        """Get the results directory."""
        return self.artifacts_dir(task_name) / "results"


def get_primitives_path(task_name: str) -> Path:
    """Get the shared primitives path for a task."""
    return ARTIFACTS_ROOT / task_name / "primitives.json"
