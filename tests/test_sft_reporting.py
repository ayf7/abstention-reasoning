import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from test_method_d import Tokenizer, TinyModel, rollouts
from test_method_e import small_model
from verl.trainer.ppo.method_d import build_prefix_batch
from verl.trainer.ppo.method_e import build_trajectory_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor


class SftReportingTests(unittest.TestCase):
    def actor(self, model):
        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.config = OmegaConf.create({"sft_micro_batch_size": 2})
        actor.device_name = "cpu"
        actor.actor_module = model
        actor.use_ulysses_sp = actor.use_fused_kernels = False
        return actor

    def batch(self, builder):
        data = rollouts([300, 20, 256])
        data.non_tensor_batch["hint_level"] = np.array([0, 2, 2], dtype=object)
        # Generated hint requests are not the input hint level.
        data.non_tensor_batch["num_hints"] = np.array([9, 9, 9])
        return builder(data, [0, 1, 0], Tokenizer(), report=True)

    def test_builders_tag_first_final_and_use_input_hints(self):
        for builder in (build_prefix_batch, build_trajectory_batch):
            with self.subTest(builder=builder.__name__):
                data = self.batch(builder)
                self.assertEqual(data.meta_info["sft_report_groups"], [
                    "hint_0", "hint_2", "position_first", "position_final",
                ])
                for position in ("first", "final"):
                    self.assertEqual(data.batch[f"sft_position_{position}"].sum().item(), 3)
                if builder is build_prefix_batch:
                    self.assertEqual(data.batch["sft_hint_level"].tolist(), [0, 0, 0, 2, 2, 2])
                    self.assertTrue(data.batch["sft_position_first"][3])
                    self.assertTrue(data.batch["sft_position_final"][3])
                else:
                    self.assertEqual(data.batch["sft_hint_level"][:, -1].tolist(), [0, 2, 2])
                    self.assertTrue((data.batch["sft_position_first"][1]
                                     & data.batch["sft_position_final"][1]).any())

    def test_method_d_reports_distinct_binary_and_token_accuracy(self):
        data = self.batch(build_prefix_batch)
        # Ignore the third trajectory, simulating distributed padding.
        data.batch["sft_weights"][-2:] = 0
        model = TinyModel()
        with torch.no_grad():
            model.logits[8] = 3  # Full-vocab prediction is not a label.
            model.logits[2] = 2  # Binary prediction is 0.
        actor = self.actor(model)
        with (
            patch("verl.workers.actor.dp_actor.get_device_id", return_value="cpu"),
            patch("torch.distributed.all_reduce"),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("verl.utils.debug.performance._get_current_mem_info", return_value=("0",) * 4),
        ):
            metrics = actor.method_d_sft(data, update=False)
        self.assertEqual(metrics["sft_groups/hint_0/accuracy"], 1)
        self.assertEqual(metrics["sft_groups/hint_0/token_accuracy"], 0)
        self.assertEqual(metrics["sft_groups/hint_0/num_examples"], 3)
        self.assertEqual(metrics["sft_groups/hint_2/accuracy"], 0)
        self.assertEqual(metrics["sft_groups/hint_2/num_examples"], 1)
        for position in ("first", "final"):
            self.assertEqual(metrics[f"sft_groups/position_{position}/accuracy"], 0.5)
            self.assertEqual(metrics[f"sft_groups/position_{position}/num_examples"], 2)

    def test_method_e_reports_positions_in_one_pass_without_token_accuracy(self):
        data = self.batch(build_trajectory_batch)
        data.batch["sft_weights"][-1] = 0
        model = small_model()
        with torch.no_grad():
            model.classification_head.weight.zero_()
            model.classification_head.bias.copy_(torch.tensor([2., 0.]))
        actor = self.actor(model)
        with (
            patch("verl.workers.actor.dp_actor.get_device_id", return_value="cpu"),
            patch("torch.distributed.all_reduce"),
            patch.object(model, "forward", wraps=model.forward) as forward,
        ):
            metrics = actor.method_e_sft(data, update=False)
        self.assertEqual(forward.call_count, 2)
        self.assertEqual(metrics["sft_groups/hint_0/accuracy"], 1)
        self.assertEqual(metrics["sft_groups/hint_2/accuracy"], 0)
        for position in ("first", "final"):
            self.assertEqual(metrics[f"sft_groups/position_{position}/accuracy"], 0.5)
        self.assertFalse(any("token_accuracy" in key for key in metrics))

    def test_training_batches_do_not_have_reporting_tensors(self):
        for builder in (build_prefix_batch, build_trajectory_batch):
            batch = builder(rollouts([20]), [1], Tokenizer())
            self.assertNotIn("sft_report_groups", batch.meta_info)
            self.assertNotIn("sft_position_first", batch.batch)

    def test_missing_and_null_hint_levels(self):
        data = rollouts([20])
        batch = build_prefix_batch(data, [1], Tokenizer(), report=True)
        self.assertIn("hint_0", batch.meta_info["sft_report_groups"])
        data.non_tensor_batch["hint_level"] = np.array([None], dtype=object)
        batch = build_trajectory_batch(data, [1], Tokenizer(), report=True)
        self.assertIn("hint_unknown", batch.meta_info["sft_report_groups"])


if __name__ == "__main__":
    unittest.main()
