"""
Framework used to standardize datasets for SFT and RL.

Standardized artifacts. Given the base directory, we will save in the
"artifacts/" directory.

[raw_dataset.jsonl]
  - index: global integer id.
  - question / answer: task-specific fields.
  - metadata: dict of task-specific info (difficulty, split label, source, etc.).

[train.jsonl / test.jsonl]
  - prompt: either a raw string or a list of chat messages (role/content).
  - cot: target text; initialized to be empty. CoT generations will modify and update
         this string in-place.
  - cot_metadata: optional dict with correct_answer (bool), length, etc.
  - index: original id.

[rl_train.parquet / rl_val.parquet]
  - prompt: recommended as a list of chat messages; if raw string, include a prompt_format flag.
  - reward_model: dict with ground_truth that the reward function needs, e.g.:
        {"ground_truth": {"solution_text": "...", "answers": [...], "solution_text_format": "..."}}
  - hint_exprs: optional list of hints for hint-enabled rollouts.
  - metadata: optional dict for analysis.
  - index: original id.
"""
from typing import Dict
from pathlib import Path

class DatasetManager:

    def __init__(self,
        path: Path,
        prompt_template: Path,
    ):
        self.path = path
        self.artifact_dir = path / "artifacts"
        self.prompt_template = prompt_template

        # should be initialized in respective function calls.
        self.cot_generation_model = None
        self.sft_base_model = None
        pass

    def create_dataset(self):
        """
        Creates the raw_dataset.jsonl. This can be done directly by pulling from
        Hugging Face, or synthetic generation.
        """
        raise NotImplementedError("create_dataset")

    def create_split(self):
        """
        Checks that [raw_dataset.jsonl] exists. Then, creates the train and test
        splits using the prompt template.
        """
        raise NotImplementedError("create_split")

    def create_generations(self,
        in_place: bool = True,
        output_file: Path | None = None
    ):
        """
        Creates CoT generations for each training and test example. Resumes at
        the first one.
        """
        raise NotImplementedError("create_generations")

    def run_sft(self):
        """
        Runs SFT on the correct CoTs done in [train.jsonl].
        """
        raise NotImplementedError("run_sft")

    def create_rl_data(self):
        """
        Checks that [train.jsonl] exists, then further splits into rl_train.parquet
        and rl_val.parquet in the artifacts directory.
        """
        raise NotImplementedError("create_rl_data")
