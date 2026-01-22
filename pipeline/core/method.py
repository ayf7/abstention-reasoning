"""Method configuration - bundles template variant + reward function + artifact paths."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .utils import model_short_name


# Default artifacts root
ARTIFACTS_ROOT = Path("artifacts")


@dataclass
class Method:
    """
    Configuration for a pipeline method.

    A method defines a consistent configuration across the full pipeline:
    - template_variant: Which template variant to use (e.g., "simple", "simple_abstention")
    - reward_function: Name of reward function for RL training
    - reward_kwargs: Additional arguments for reward function
    - assistant_prefix: Prefix for assistant responses (used in SFT and RL)

    Methods also provide auto-derived artifact paths based on task and method name.
    """

    name: str
    template_variant: str
    reward_function: str = "compute_score"
    reward_kwargs: dict[str, Any] = field(default_factory=dict)
    assistant_prefix: str | None = None  # If None, use task's default

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
            config_path = Path(f"pipeline/configs/methods/{task_name}/{name_or_path}.yaml")
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
            assistant_prefix=data.get("assistant_prefix"),
        )

    @staticmethod
    def list_methods(task_name: str) -> list[str]:
        """List available methods for a task."""
        methods_dir = Path(f"pipeline/configs/methods/{task_name}")
        if not methods_dir.exists():
            return []
        return [p.stem for p in methods_dir.glob("*.yaml")]

    def get_template_path(self, task_name: str, split: str) -> Path:
        """Get the template path for a given split."""
        return Path(f"pipeline/tasks/{task_name}/templates/{self.template_variant}/{split}.txt")

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

    def primitives_path(self, task_name: str) -> Path:
        """Get the primitives path (shared, not method-specific)."""
        return ARTIFACTS_ROOT / task_name / "primitives.json"

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

    def sft_model_path(self, task_name: str) -> Path:
        """Get the SFT model path."""
        return self.models_dir(task_name) / "sft"

    def rl_model_path(self, task_name: str) -> Path:
        """
        Get the RL model path.

        First checks for models/rl directory. If not found, searches for
        the latest checkpoint matching *_rl_step* pattern (highest step number).
        """
        import re

        models_dir = self.models_dir(task_name)
        default_path = models_dir / "rl"

        # If default rl/ directory exists, use it
        if default_path.exists():
            return default_path

        # Otherwise, find latest *_rl_step* checkpoint
        if models_dir.exists():
            rl_checkpoints = []
            for d in models_dir.iterdir():
                if d.is_dir() and "_rl_step" in d.name:
                    match = re.search(r"_rl_step(\d+)", d.name)
                    if match:
                        step_num = int(match.group(1))
                        rl_checkpoints.append((step_num, d))

            if rl_checkpoints:
                # Return the one with highest step number
                latest = max(rl_checkpoints, key=lambda x: x[0])
                return latest[1]

        # Fallback to default path (will error if not found)
        return default_path

    def classifier_model_path(self, task_name: str) -> Path:
        """Get the classifier model path."""
        return self.models_dir(task_name) / "classifier"

    def results_dir(self, task_name: str) -> Path:
        """Get the results directory."""
        return self.artifacts_dir(task_name) / "results"

    def results_path(self, task_name: str, model_name: str) -> Path:
        """Get the results path for a specific model."""
        model_slug = model_short_name(model_name)
        return self.results_dir(task_name) / f"eval_{model_slug}.json"


def get_primitives_path(task_name: str) -> Path:
    """Get the shared primitives path for a task."""
    return ARTIFACTS_ROOT / task_name / "primitives.json"
