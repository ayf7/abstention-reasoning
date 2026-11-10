import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from datasets import Dataset

from pipeline.commands.sft_data import prepare_multisample_sft
from pipeline.commands.training import train_sft
from pipeline.core.io import load_json, save_json
from pipeline.tasks import get_task


class ChatTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            for m in messages
        ) + "<|im_start|>assistant\n"

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return list(map(ord, text))


class MultisampleSftTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "tree.json"
        self.output = self.root / "sft.json"
        self.tokenizer = ChatTokenizer()
        patcher = patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=self.tokenizer,
        )
        self.load_tokenizer = patcher.start()
        self.addCleanup(patcher.stop)
        self.rows = [{
            "index": 9919,
            "label": label,
            "distance": 0.08322496749024708,
            "prefix_tokens": 128,
            "assistant_prefix": "<think> Let me solve this step by step.",
            "prompt": [
                {"role": "system", "content": "Solve the puzzle."},
                {"role": "user", "content": "Make 104 using [34, 11, 2, 28, 3, 1]."},
                {"role": "assistant", "content": "<think> Let me solve this step by step. Looking at"},
            ],
            "generation": str(label),
            "samples": [{"generation": "DO NOT LEAK THE CONTINUATION", "correct": bool(label)}],
            "correct": False,
        } for label in (0, 1)]

    def prepare(self):
        save_json(self.source, self.rows, max_shard_bytes=1)
        prepare_multisample_sft(self.source, self.output, "training-model")
        return load_json(self.output)

    def test_all_shards_preserve_both_labels_and_exact_context(self):
        original = copy.deepcopy(self.rows)
        rows = self.prepare()
        self.assertEqual(self.rows, original)
        self.assertFalse(self.source.exists())
        self.assertEqual([row["generation"] for row in rows], ["0", "1"])
        self.assertTrue(all(row["correct"] for row in rows))
        expected = (
            "<|im_start|>system\nSolve the puzzle.<|im_end|>\n"
            "<|im_start|>user\nMake 104 using [34, 11, 2, 28, 3, 1].<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think> Let me solve this step by step. Looking at<continue>"
        )
        self.assertEqual(rows[0]["prompt"], expected)
        self.assertNotIn("samples", rows[0])
        self.assertEqual(rows[0]["distance"], self.rows[0]["distance"])
        self.load_tokenizer.assert_called_once_with("training-model")
        for task_name in ("math", "sql"):
            self.assertEqual(len(get_task(task_name).filter_for_sft(rows)), 2)

    def test_root_prefix_is_not_duplicated(self):
        self.rows[0]["prompt"][-1]["content"] = self.rows[0]["assistant_prefix"]
        rows = self.prepare()
        self.assertTrue(rows[0]["prompt"].endswith(
            "assistant\n<think> Let me solve this step by step.<continue>"
        ))
        self.assertEqual(rows[0]["prompt"].count("<think>"), 1)

    def test_rejects_invalid_labels_prefixes_and_overwriting_source(self):
        for label in (None, -1, 2, "0", True):
            with self.subTest(label=label):
                self.rows[0]["label"] = label
                save_json(self.source, self.rows)
                with self.assertRaisesRegex(ValueError, "integer label"):
                    prepare_multisample_sft(self.source, self.output, "model")
        self.rows[0]["label"] = 0
        self.rows[0]["prompt"].pop()
        save_json(self.source, self.rows)
        with self.assertRaisesRegex(ValueError, "assistant prefix"):
            prepare_multisample_sft(self.source, self.output, "model")
        with self.assertRaisesRegex(ValueError, "differ"):
            prepare_multisample_sft(self.source, self.source, "model")

    def test_train_sft_keeps_labels_and_masks_context_in_both_paths(self):
        prepared = self.prepare()
        for masked in (False, True):
            with self.subTest(masked=masked):
                config = Mock()
                trainer = Mock()
                method = Mock(mask_response_tokens=masked, nested_request=False)
                method.dataset_path.return_value = self.root / "missing_eval.json"
                with (
                    patch.dict(sys.modules, {"trl": SimpleNamespace(
                        SFTConfig=config, SFTTrainer=trainer,
                    )}),
                    patch("pipeline.commands.training.Method.load", return_value=method),
                    patch("pipeline.commands.training._finalize_training_run"),
                ):
                    train_sft(
                        task_name="countdown", base_model="model", method_name="baseline",
                        dataset_path=self.output, output_path=self.root / "model",
                        report_to="none",
                    )
                self.assertTrue(config.call_args.kwargs["completion_only_loss"])
                dataset = trainer.call_args.kwargs["train_dataset"]
                self.assertIsInstance(dataset, Dataset)
                self.assertEqual(len(dataset), 2)
                for i, row in enumerate(dataset):
                    if masked:
                        context_size = len(prepared[i]["prompt"])
                        self.assertEqual(row["completion_mask"], [0] * context_size + [1])
                        self.assertEqual(row["input_ids"][-1], ord(str(i)))
                    else:
                        self.assertEqual(row, {
                            "prompt": prepared[i]["prompt"], "completion": str(i),
                        })

    def test_existing_chat_sft_format_and_loss_are_unchanged(self):
        self.rows[0]["correct"] = True
        self.rows[0]["generation"] = " the solution"
        save_json(self.source, self.rows)
        config, trainer = Mock(), Mock()
        with (
            patch.dict(sys.modules, {"trl": SimpleNamespace(
                SFTConfig=config, SFTTrainer=trainer,
            )}),
            patch("pipeline.commands.training._finalize_training_run"),
        ):
            train_sft(
                task_name="countdown", base_model="model", dataset_path=self.source,
                output_path=self.root / "model", report_to="none",
            )
        self.assertFalse(config.call_args.kwargs["completion_only_loss"])
        dataset = trainer.call_args.kwargs["train_dataset"]
        self.assertEqual(len(dataset), 1)
        self.assertEqual(
            dataset[0]["completion"],
            self.rows[0]["prompt"][-1]["content"] + " the solution",
        )

    def test_cli_passes_explicit_paths_and_training_model(self):
        from pipeline.__main__ import main

        with (
            patch.object(sys, "argv", [
                "pipeline", "prepare_multisample_sft", "--dataset", str(self.source),
                "--output", str(self.output), "--base-model", "training-model",
            ]),
            patch("pipeline.commands.prepare_multisample_sft") as prepare,
        ):
            main()
        prepare.assert_called_once_with(
            dataset_path=self.source, output_path=self.output, base_model="training-model",
        )


if __name__ == "__main__":
    unittest.main()
