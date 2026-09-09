"""Method configuration - bundles template variant + reward function + artifact paths."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


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
    - template_variant: Which template variant to use (e.g., "simple", "hint")
    - reward_function: Name of reward function for RL training
    - reward_kwargs: Additional arguments for reward function
    - multi_turn: Enable multi-turn hint generation in RL (default: False)
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
    mask_response_tokens: bool = False  # Mask <response>...</response> in SFT
    max_hints: int | None = None  # Maximum number of hints to give during RL rollout (None = unlimited)
    hint_transition: bool = True  # Splice a canned "I'm stuck..." phrase before each
                                  # forced hint request during SFT data generation. False
                                  # cuts the CoT silently instead, so nothing before
                                  # <request></request> telegraphs the request.
    nested_request: bool = False  # Keep <request></request> inside the <think> block
                                  # instead of after it, so </think> stays the single
                                  # irreversible commit point right before <answer>.

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
            name_or_path: Either a method name (e.g., "baseline", "method_b")
                          or a path to a YAML config file
            task_name: Task name (used to find config in standard location)

        Returns:
            Method instance

        Lookup order:
            1. If name_or_path is a file path, load directly
            2. Otherwise, look in pipeline/configs/methods/{task}/{name}.yaml

        Either way the method's name is the config file's stem.
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

        # The filename is the name, and the only source of it. It decides the
        # templates, the __suffix on every prompt and dataset file, and the
        # model group directory, so a `name:` key free to disagree with the
        # file it sits in would be a second source of truth for all three.
        return cls(
            name=config_path.stem,
            template_variant=data["template_variant"],
            reward_function=data.get("reward_function", "compute_score"),
            reward_kwargs=data.get("reward_kwargs", {}),
            multi_turn=data.get("multi_turn", False),
            max_turns=data.get("max_turns", 6),
            mask_response_tokens=data.get("mask_response_tokens", False),
            max_hints=data.get("max_hints"),
            hint_transition=data.get("hint_transition", True),
            nested_request=data.get("nested_request", False),
        )

    @staticmethod
    def list_methods(task_name: str) -> list[str]:
        """List available methods for a task."""
        methods_dir = CONFIGS_ROOT / task_name
        if not methods_dir.exists():
            return []
        return sorted(p.stem for p in methods_dir.glob("*.yaml"))

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
    def task_dir(self, task_name: str) -> Path:
        """Root of a task's artifacts. Method-independent: prompts, datasets and
        models all live in shared directories and are told apart by name, not by
        sitting under a per-method subtree."""
        return ARTIFACTS_ROOT / task_name

    # -- prompts and datasets -------------------------------------------------

    def artifact_stem(self, split: str, desc: str | None = None) -> str:
        """`{split}__{method}` (+ `_{desc}`), the shared stem for a prompt file
        and the dataset generated from it."""
        stem = f"{split}__{self.name}"
        return f"{stem}_{desc}" if desc else stem

    def prompts_dir(self, task_name: str) -> Path:
        return self.task_dir(task_name) / "prompts"

    def prompts_path(self, task_name: str, split: str, desc: str | None = None) -> Path:
        ext = ".parquet" if split.startswith("rl") else ".json"
        return self.prompts_dir(task_name) / f"{self.artifact_stem(split, desc)}{ext}"

    def datasets_dir(self, task_name: str) -> Path:
        return self.task_dir(task_name) / "datasets"

    def dataset_path(self, task_name: str, split: str, desc: str | None = None) -> Path:
        return self.datasets_dir(task_name) / f"{self.artifact_stem(split, desc)}.json"

    # -- models ---------------------------------------------------------------

    def models_dir(self, task_name: str) -> Path:
        return self.task_dir(task_name) / "models"

    # Directory suffix per training stage. RL output is the finished model for
    # a method, so it lands in `_models`; `_sft` holds the intermediate it was
    # initialized from. The stage is still spelled "rl" everywhere in the code
    # and on the command line -- only the directory reads as the result.
    GROUP_SUFFIX = {"sft": "sft", "rl": "models"}

    def group_dir(self, task_name: str, stage: str) -> Path:
        """`models/{method}_{sft,models}` -- e.g. models/baseline_sft, models/method_b_models."""
        try:
            suffix = self.GROUP_SUFFIX[stage]
        except KeyError:
            raise ValueError(
                f"stage must be one of {sorted(self.GROUP_SUFFIX)}, got {stage!r}"
            ) from None
        return self.models_dir(task_name) / f"{self.name}_{suffix}"

    def run_dir(self, task_name: str, stage: str, run_id: str) -> Path:
        """One training run: `models/{method}_{sft,models}/{run_id}`.

        run_id is the directory name verbatim and is always required. Nothing
        derives it from the base checkpoint: the convention (`1.5b`, `4b`,
        `4b-instruct`, `1.5b__extend-quad-a0.5`) is a naming decision, not a
        fact about the model, so a guessed name would only drift from it.
        """
        if not run_id:
            raise ValueError(
                f"--run-id is required: it names the directory under "
                f"{self.group_dir(task_name, stage).relative_to(ARTIFACTS_ROOT.parent)}."
            )
        return self.group_dir(task_name, stage) / run_id

    def _ensure_run_dir(self, run_dir: Path, task_name: str) -> Path:
        """Create a run directory, or resolve it to external storage.

        With EXTERNAL_MODELS_ROOT set, a run that does not exist yet is created
        there and symlinked in, so new runs land on shared storage even when
        older ones are local.
        """
        # A convenience alias (models/baseline_rl/4b -> 4b__stdnorm) names a
        # sibling run in the same directory, so a short name resolves without
        # renaming anything. Training through one would write into the run it
        # points at and destroy it. Such aliases are relative and have no "/";
        # the external-storage links created below are absolute and remain
        # valid resume targets.
        if run_dir.is_symlink():
            target = os.readlink(run_dir)
            if "/" not in target:
                raise FileExistsError(
                    f"'{run_dir.name}' is a convenience symlink to sibling run "
                    f"'{target}', not a run directory of its own. Training here "
                    f"would overwrite that run.\n"
                    f"  Use --run-id {target} to train or resume it, "
                    f"or pick a new run id."
                )
            return run_dir

        if run_dir.exists():
            return run_dir

        if EXTERNAL_MODELS_ROOT is not None:
            rel_to_models = run_dir.relative_to(self.models_dir(task_name))
            external_path = EXTERNAL_MODELS_ROOT / task_name / rel_to_models
            external_path.mkdir(parents=True, exist_ok=True)
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            run_dir.symlink_to(external_path)
            print(f"Created symlink: {run_dir} -> {external_path}")
        else:
            run_dir.mkdir(parents=True, exist_ok=True)

        return run_dir

    def sft_run_dir(self, task_name: str, run_id: str) -> Path:
        return self.run_dir(task_name, "sft", run_id)

    def rl_run_dir(self, task_name: str, run_id: str) -> Path:
        return self.run_dir(task_name, "rl", run_id)

    def ensure_sft_run_dir(self, task_name: str, run_id: str) -> Path:
        return self._ensure_run_dir(self.sft_run_dir(task_name, run_id), task_name)

    def ensure_rl_run_dir(self, task_name: str, run_id: str) -> Path:
        return self._ensure_run_dir(self.rl_run_dir(task_name, run_id), task_name)

    def sft_model_path(self, task_name: str, run_id: str) -> Path:
        return self.sft_run_dir(task_name, run_id) / "model"

    def rl_model_path(self, task_name: str, run_id: str) -> Path:
        return self.rl_run_dir(task_name, run_id) / "model"

    def rl_checkpoints_dir(self, task_name: str, run_id: str) -> Path:
        return self.rl_run_dir(task_name, run_id) / "checkpoints"

    def rl_rollouts_dir(self, task_name: str, run_id: str) -> Path:
        return self.rl_run_dir(task_name, run_id) / "rollouts"

    # -- evaluations ----------------------------------------------------------

    def evals_dir(self, task_name: str, stage: str, run_id: str) -> Path:
        """`models/{method}_{sft,models}/{run_id}/evals` -- results live inside the
        run that produced them, not in a task-wide results/ pool."""
        return self.run_dir(task_name, stage, run_id) / "evals"

    def eval_path(self, task_name: str, stage: str, run_id: str,
                  split: str, suffix: str = "") -> Path:
        """`.../evals/{split}{suffix}.json`. The run directory already carries
        the model identity, so the filename only has to say which split and
        under what deviation from the default settings."""
        return self.evals_dir(task_name, stage, run_id) / f"{split}{suffix}.json"


def problems_dir(task_name: str) -> Path:
    """`artifacts/{task}/problems` -- raw problems, shared by every method."""
    return ARTIFACTS_ROOT / task_name / "problems"


def get_primitives_path(task_name: str) -> Path:
    """Get the shared primitives path for a task."""
    return problems_dir(task_name) / "primitives.json"
