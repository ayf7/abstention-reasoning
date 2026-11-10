import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline.commands.tree_multisample import generate_tree_multisample
from pipeline.core.method import DATA_ROOT, MODELS_ROOT, Method


MODULE = "pipeline.commands.tree_multisample"


class CharacterTokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return list(map(ord, text))

    def decode(self, tokens, **kwargs):
        return "".join(map(chr, tokens))


class TreeMultisampleTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name) / "tree.json"
        self.prompts = [{
            "index": 7,
            "variant": "test",
            "ground_truth": {"answer": "GOOD"},
            "source_index": 31,
            "prompt": [
                {"role": "user", "content": "Solve"},
                {"role": "assistant", "content": "<think>\n"},
            ],
        }]
        self.method = Mock(nested_request=False, multi_turn=False)
        self.method.formatted_path.return_value = Path("prompts.json")
        self.method.dataset_path.return_value = self.output
        self.task = Mock(system_message=None)
        self.task.check_correctness.side_effect = (
            lambda primitive, text: (text.endswith("GOOD"), {"predicted_answer": text[-4:]})
        )
        patches = [
            patch(f"{MODULE}.Method.load", return_value=self.method),
            patch(f"{MODULE}.get_task", return_value=self.task),
            patch(f"{MODULE}.load_prompts", return_value=self.prompts),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=CharacterTokenizer()),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        self.calls = []
        self.trajectory = "a b\nc!"

        def generate(prompts, count):
            self.calls.append((prompts, count))
            if len(self.calls) == 1:
                return [[{"text": self.trajectory}] for _ in prompts]
            return [[{
                "text": "GOOD" if i % 2 == 0 else "BAD",
                "finish_reason": "stop",
                "token_count": 1,
            } for i in range(count)] for _ in prompts]

        self.generate = generate
        self.generator = Mock()
        self.configs = []

        def build(config):
            self.configs.append(config)
            self.generator.generate_batched.side_effect = lambda prompts: self.generate(
                prompts, config.num_samples
            )
            return self.generator

        patcher = patch(f"{MODULE}.Generator", side_effect=build)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_generation(self, **kwargs):
        options = dict(
            task_name="countdown", model_name="solver", method_name="baseline",
            split="sft_train", stride=2, num_points=10, num_samples=4,
        )
        options.update(kwargs)
        result = generate_tree_multisample(**options)
        self.assertEqual(result, self.output)
        return json.loads(result.read_text())

    def test_each_prefix_has_individual_targets_and_full_text_correctness(self):
        records = self.run_generation()

        self.assertEqual(len(records), 16)
        self.assertEqual([count for _, count in self.calls], [1, 4, 4, 4, 4])
        for point, position in enumerate((0, 2, 4, 6)):
            group = records[point * 4:(point + 1) * 4]
            prefix = "<think>\n" + self.trajectory[:position]
            self.assertEqual([record["label"] for record in group], [1, 0, 1, 0])
            self.assertEqual([record["generation"] for record in group], ["1", "0", "1", "0"])
            self.assertEqual([record["sample_index"] for record in group], [0, 1, 2, 3])
            for record in group:
                self.assertEqual(record["prompt"][-1]["content"], prefix)
                self.assertEqual(record["prefix_tokens"], position)
                self.assertEqual(record["trajectory_tokens"], 6)
                self.assertEqual(record["source_index"], 31)
                self.assertEqual(record["ground_truth"], {"answer": "GOOD"})
                self.assertNotIn("correct", record)
                sample = record["samples"][0]
                self.assertEqual(sample["generation"], prefix + sample["continuation"])
        checked_texts = [call.args[1] for call in self.task.check_correctness.call_args_list]
        self.assertEqual(checked_texts, [
            record["samples"][0]["generation"] for record in records
        ])

    def test_random_positions_are_distinct_seeded_and_support_one_sample(self):
        first = self.run_generation(num_points=2, num_samples=1, seed=13)
        positions = [record["prefix_tokens"] for record in first]
        self.assertEqual(len(set(positions)), 2)
        self.assertTrue(set(positions) <= {0, 2, 4, 6})
        self.assertEqual(len(self.calls), 3)

        self.calls.clear()
        second = self.run_generation(num_points=2, num_samples=1, seed=13)
        self.assertEqual(first, second)

    def test_short_empty_trajectory_without_assistant_prefix(self):
        self.trajectory = ""
        self.prompts[0]["prompt"].pop()
        records = self.run_generation(num_samples=1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["prefix_tokens"], 0)
        self.assertEqual(records[0]["distance"], 0.0)
        self.assertEqual(records[0]["prompt"][-1], {"role": "assistant", "content": ""})

    def test_examples_keep_their_own_trajectories_and_available_points(self):
        self.prompts.append({
            "index": 8, "variant": "short", "ground_truth": {"answer": "GOOD"},
            "prompt": [{"role": "user", "content": "Another problem"}],
        })
        generate = self.generate

        def different_lengths(prompts, count):
            results = generate(prompts, count)
            if len(self.calls) == 1:
                results[1][0]["text"] = "x"
            return results

        self.generate = different_lengths
        records = self.run_generation(num_samples=2)
        long_records = [record for record in records if record["index"] == 7]
        short_records = [record for record in records if record["index"] == 8]
        self.assertEqual(len(long_records), 8)
        self.assertEqual(len(short_records), 2)
        self.assertEqual({record["prefix_tokens"] for record in short_records}, {0})
        self.assertEqual({record["trajectory_tokens"] for record in short_records}, {1})
        self.assertEqual([len(prompts) for prompts, _ in self.calls], [2, 2, 1, 1, 1])

    def test_async_matches_sync_and_closes_engine(self):
        expected = self.run_generation(num_points=2)
        self.calls.clear()
        generator = Mock()

        async def generate(prompts, num_samples):
            return self.generate(prompts, num_samples)

        generator.generate_async = generate
        with patch(f"{MODULE}.AsyncGenerator", return_value=generator):
            actual = self.run_generation(num_points=2, use_async=True)
        self.assertEqual(actual, expected)
        generator.close.assert_called_once_with()

    def test_async_failure_keeps_previous_round_and_closes_engine(self):
        generator = Mock()

        async def generate(prompts, num_samples):
            if len(self.calls) == 2:
                raise RuntimeError("generation failed")
            return self.generate(prompts, num_samples)

        generator.generate_async = generate
        with patch(f"{MODULE}.AsyncGenerator", return_value=generator):
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                self.run_generation(use_async=True)
        self.assertEqual(len(json.loads(self.output.read_text())), 4)
        generator.close.assert_called_once_with()

    def test_rejects_missing_samples_instead_of_silently_dropping_records(self):
        self.generator.generate_batched.side_effect = None
        self.generator.generate_batched.return_value = []
        with patch(f"{MODULE}.Generator", return_value=self.generator):
            with self.assertRaisesRegex(ValueError, "Expected 1 samples"):
                self.run_generation()

    def test_validates_sampling_arguments(self):
        for name in ("stride", "num_points", "num_samples", "batch_size", "max_new_tokens"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    self.run_generation(**{name: 0})

    def test_resolves_rl_model_and_parquet_prompts(self):
        self.method.formatted_path.return_value = Path("prompts.parquet")
        self.method.load_template.return_value = "Problem: {problem}"
        self.method.rl_model_path.return_value = Path("models/solver")
        self.run_generation(model_name="rl", run_id="run", data_name="puzzles")
        self.method.rl_model_path.assert_called_once_with("puzzles", "run")
        self.method.load_template.assert_called_once_with("countdown", "rl")
        self.method.dataset_path.assert_called_once_with(
            "puzzles", "sft_train", desc="tree_multisample__run"
        )

    def test_data_name_and_models_name_resolve_real_model_paths(self):
        method = Method(name="baseline", template_variant="baseline")
        for stage in ("sft", "rl"):
            for models_name in (None, "shared_models"):
                with self.subTest(stage=stage, models_name=models_name):
                    self.calls.clear()
                    with patch(f"{MODULE}.Method.load", return_value=method), patch(
                        f"{MODULE}.load_prompts", return_value=self.prompts
                    ) as loader:
                        self.run_generation(
                            task_name="math", data_name="math_o1", models_name=models_name,
                            model_name=stage, run_id="run", output_path=self.output,
                        )
                    expected = MODELS_ROOT / (models_name or "math_o1") / f"baseline_{stage}" / "run" / "model"
                    self.assertEqual(self.configs[-1].model_name, str(expected))
                    self.assertEqual(
                        loader.call_args.args[0],
                        DATA_ROOT / "math_o1" / "problems_with_format" / "sft_train__baseline.json",
                    )

    def test_explicit_prompts_and_output_work_without_method(self):
        self.run_generation(
            method_name=None, prompts_path=Path("custom.json"),
            output_path=self.output, verbose=True, no_hints=True,
        )
        self.method.formatted_path.assert_not_called()
        self.assertTrue(self.configs[-1].verbose)
        self.assertTrue(self.configs[-1].ban_hint_requests)
        self.assertFalse(self.configs[-1].nested_request)

    def test_methodless_generation_requires_explicit_paths(self):
        with self.assertRaisesRegex(ValueError, "--method or --prompts"):
            self.run_generation(method_name=None)
        with self.assertRaisesRegex(ValueError, "--method or --output"):
            self.run_generation(method_name=None, prompts_path=Path("custom.json"))

    def test_multiturn_replicates_each_prefix_and_its_ground_truth(self):
        self.method.multi_turn = True
        hint_calls = []

        def hints(prompts, truths):
            hint_calls.append((prompts, truths))
            return self.generate(prompts, 1)

        self.generator.generate_with_hints_batched.side_effect = hints
        records = self.run_generation(num_points=1, num_samples=3)
        self.assertEqual([len(prompts) for prompts, _ in hint_calls], [1, 3])
        self.assertEqual(hint_calls[1][1], [self.prompts[0]["ground_truth"]] * 3)
        self.assertEqual(len(records), 3)
        self.generator.generate_batched.assert_not_called()

    def test_async_multiturn_can_be_explicitly_enabled(self):
        generator = Mock()
        hint_calls = []

        async def hints(prompts, truths):
            hint_calls.append((prompts, truths))
            return self.generate(prompts, 1)

        generator.generate_with_hints_async = hints
        with patch(f"{MODULE}.AsyncGenerator", return_value=generator):
            records = self.run_generation(
                multi_turn=True, use_async=True, num_points=1, num_samples=2,
            )
        self.assertEqual(len(records), 2)
        self.assertEqual([len(prompts) for prompts, _ in hint_calls], [1, 2])
        generator.close.assert_called_once_with()

    def test_function_accepts_all_evaluate_parameters_with_same_defaults(self):
        from pipeline.commands.inference import evaluate

        actual = inspect.signature(generate_tree_multisample).parameters
        for name, parameter in inspect.signature(evaluate).parameters.items():
            with self.subTest(parameter=name):
                self.assertIn(name, actual)
                self.assertEqual(actual[name].default, parameter.default)

    def test_cli_matches_all_evaluate_options_and_defaults(self):
        from pipeline.__main__ import main

        common = [
            "--task", "math", "--model", "rl", "--method", "baseline",
            "--run-id", "run",
        ]
        explicit = [
            "--data-name", "math_o1", "--models-name", "shared_models",
            "--split", "rl_ver_train", "--prompts", "custom.json",
            "--output", str(self.output), "--batch-size", "3",
            "--max-new-tokens", "100", "--temperature", "0.5",
            "--top-p", "0.8", "--num-samples", "4", "--tensor-parallel-size", "2",
            "--data-parallel-size", "3", "--gpu-memory-utilization", "0.6",
            "--verbose", "--multi-turn", "--async", "--seed", "9", "--no-hints",
        ]
        for options in (common, common + explicit):
            with self.subTest(options=options):
                with patch("sys.argv", ["pipeline", "evaluate", *options]), patch(
                    "pipeline.commands.evaluate"
                ) as evaluate:
                    main()
                with patch("sys.argv", [
                    "pipeline", "generate_tree_multisample", *options,
                    "--stride", "128", "--num-points", "5",
                ]), patch("pipeline.commands.generate_tree_multisample") as generate:
                    main()
                forwarded = dict(generate.call_args.kwargs)
                forwarded.pop("stride")
                forwarded.pop("num_points")
                self.assertEqual(forwarded, evaluate.call_args.kwargs)

    def test_cli_forwards_sampling_options(self):
        from pipeline.__main__ import main

        with patch("sys.argv", [
            "pipeline", "generate_tree_multisample", "--task", "countdown",
            "--method", "baseline", "--model", "rl", "--run-id", "run",
            "--split", "sft_train", "--stride", "128", "--num-points", "5",
            "--num-samples", "8", "--async", "--output", str(self.output),
        ]), patch("pipeline.commands.generate_tree_multisample") as command:
            main()
        self.assertEqual(command.call_args.kwargs["stride"], 128)
        self.assertEqual(command.call_args.kwargs["num_points"], 5)
        self.assertEqual(command.call_args.kwargs["num_samples"], 8)
        self.assertTrue(command.call_args.kwargs["use_async"])
        self.assertEqual(command.call_args.kwargs["output_path"], self.output)


if __name__ == "__main__":
    unittest.main()
