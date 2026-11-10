import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.method_d import (
    build_prefix_batch,
    classification_statistics,
    classification_tokens,
    is_sft_step,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.workers.actor.dp_actor import DataParallelPPOActor


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        return {"<complete>": [6, 7], "0": [2], "1": [3]}[text]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids.tolist()))

    def batch_decode(self, ids, skip_special_tokens=True):
        return [self.decode(row) for row in ids]


def rollouts(lengths):
    responses = torch.zeros(len(lengths), max(lengths) + 1, dtype=torch.long)
    mask = torch.zeros(len(lengths), 3 + responses.shape[1], dtype=torch.long)
    prompts = torch.tensor([[0, 4, 5]] * len(lengths))
    mask[:, 1:3] = 1
    for i, length in enumerate(lengths):
        responses[i, :length] = 8
        responses[i, length] = 9
        mask[i, 3 : 3 + length + 1] = 1
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat([prompts, responses], -1),
            "attention_mask": mask,
            "position_ids": (mask.cumsum(-1) - 1).clamp_min(0),
        }
    )


class MethodDPrefixTests(unittest.TestCase):
    def test_strided_prefixes_and_full_trajectory_keep_each_sample_label(self):
        data = build_prefix_batch(rollouts([300, 256, 20]), [0, 1, 0], Tokenizer())
        self.assertEqual(data.batch["attention_mask"].sum(-1).tolist(), [132, 260, 304, 132, 260, 24])
        self.assertEqual(data.batch["sft_targets"].tolist(), [2, 2, 2, 3, 3, 2])
        for ids, mask in zip(data.batch["input_ids"], data.batch["attention_mask"], strict=True):
            context = ids[mask.bool()].tolist()
            self.assertEqual(context[:2], [4, 5])
            self.assertEqual(context[-2:], [6, 7])
            self.assertEqual(set(context[2:-2]), {8})
            self.assertNotIn(9, context)

    def test_short_and_empty_outputs_still_have_one_full_example(self):
        data = build_prefix_batch(rollouts([0, 1]), [0, 1], Tokenizer())
        self.assertEqual(data.batch["attention_mask"].sum(-1).tolist(), [4, 5])

    def test_invalid_labels_tokens_stride_and_context_fail_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            build_prefix_batch(rollouts([10]), [0.1], Tokenizer())
        with self.assertRaisesRegex(ValueError, "one correctness"):
            build_prefix_batch(rollouts([10]), [], Tokenizer())
        with self.assertRaisesRegex(ValueError, "positive"):
            build_prefix_batch(rollouts([10]), [0], Tokenizer(), stride=0)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_prefix_batch(rollouts([128]), [0], Tokenizer(), max_model_len=132)
        tokenizer = Tokenizer()
        tokenizer.encode = lambda text, **kwargs: [1, 2]
        with self.assertRaisesRegex(ValueError, "single-token"):
            classification_tokens(tokenizer)

    def test_full_vocab_accuracy_differs_from_binary_accuracy_and_ignores_padding(self):
        logits = torch.tensor([[5.0, 0.0, 1.0, 3.0], [0.0, 0.0, 4.0, 1.0], [8.0, 0.0, 1.0, 0.0]], requires_grad=True)
        targets = torch.tensor([3, 2, 3])
        stats = classification_statistics(logits, targets, torch.tensor([1.0, 1.0, 0.0]), [2, 3])
        self.assertEqual(stats["token_accuracy"].item(), 1)
        self.assertEqual(stats["accuracy"].item(), 2)
        self.assertEqual(stats["num_examples"].item(), 2)
        expected = torch.nn.functional.cross_entropy(logits[:2], targets[:2], reduction="sum")
        torch.testing.assert_close(stats["loss"], expected)
        stats["loss"].backward()
        self.assertEqual(logits.grad[2].abs().sum().item(), 0)

    def test_alternation_uses_persistent_step_number(self):
        self.assertEqual([is_sft_step(step, True) for step in range(1, 5)], [True, False, True, False])
        self.assertTrue(is_sft_step(26, False))
        self.assertFalse(is_sft_step(26, True))


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(10))
        self.contexts = []

    def forward(self, input_ids, **kwargs):
        self.contexts.append(input_ids.detach().clone())
        return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, -1))


class MethodDActorTests(unittest.TestCase):
    def test_only_next_token_trained_and_padded_examples_do_not_affect_update(self):
        data = build_prefix_batch(rollouts([20, 30, 40]), [1, 1, 0], Tokenizer())
        data.batch["sft_weights"][-1] = 0
        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.config = OmegaConf.create({"sft_micro_batch_size": 1})
        actor.use_ulysses_sp = False
        actor.use_fused_kernels = False
        actor.device_name = "cpu"
        actor.actor_module = TinyModel()
        actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.1)

        def step():
            actor.actor_optimizer.step()
            return torch.tensor(0.0)

        actor._optimizer_step = step
        with (
            patch("verl.workers.actor.dp_actor.get_device_id", return_value="cpu"),
            patch("torch.distributed.all_reduce"),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("verl.utils.debug.performance._get_current_mem_info", return_value=("0",) * 4),
        ):
            before = actor.method_d_sft(data, update=False)
            self.assertEqual(before["sft_num_examples"], 2)
            self.assertAlmostEqual(before["sft_loss"], np.log(10), places=5)
            actor.method_d_sft(data, update=True)
            after = actor.method_d_sft(data, update=False)
        self.assertLess(after["sft_loss"], before["sft_loss"])
        self.assertEqual(after["sft_token_accuracy"], 1)
        self.assertEqual(after["sft_accuracy"], 1)
        # The two real labels are 1; the masked 0 must contribute no gradient.
        self.assertGreater(actor.actor_module.logits[3].item(), 0)
        self.assertLess(actor.actor_module.logits[2].item(), 0)
        for context in actor.actor_module.contexts:
            self.assertEqual(context[0, -2:].tolist(), [6, 7])


class MethodDTrainerTests(unittest.TestCase):
    def trainer(self):
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.config = OmegaConf.create(
            {
                "trainer": {"method_d": True, "sft_stride": 128},
                "actor_rollout_ref": {"rollout": {"max_model_len": 4096}},
            }
        )
        trainer.tokenizer = Tokenizer()
        trainer.actor_rollout_wg = Mock(world_size=4)
        trainer.actor_rollout_wg.update_actor.return_value = DataProto(meta_info={"metrics": {"sft_accuracy": [1.0]}})
        trainer.actor_rollout_wg.evaluate_method_d.return_value = DataProto(
            meta_info={"metrics": {"sft_accuracy": [0.5]}}
        )
        return trainer

    def test_prefix_dispatch_masks_distributed_padding_and_routes_eval_without_updates(self):
        trainer = self.trainer()
        self.assertEqual(trainer._method_d_sft(rollouts([20]), [1], True)["sft_accuracy"], 1)
        sent = trainer.actor_rollout_wg.update_actor.call_args.args[0]
        self.assertEqual(len(sent), 4)
        self.assertEqual(sent.batch["sft_weights"].tolist(), [1, 0, 0, 0])
        self.assertTrue(sent.meta_info["method_d_sft"])
        self.assertEqual(trainer._method_d_sft(rollouts([20]), [0], False)["sft_accuracy"], 0.5)
        self.assertEqual(trainer.actor_rollout_wg.update_actor.call_count, 1)
        trainer.actor_rollout_wg.evaluate_method_d.assert_called_once()
        with self.assertRaisesRegex(ValueError, "correct"):
            trainer._method_d_sft(rollouts([20]), None, True)

    def test_fit_alternates_real_sft_and_existing_rl_paths_and_preserves_validation(self):
        trainer = self.trainer()
        trainer.config = OmegaConf.create(
            {
                "trainer": {
                    "method_d": True,
                    "interleave_rl": True,
                    "sft_stride": 128,
                    "logger": [],
                    "project_name": "test",
                    "experiment_name": "test",
                    "val_before_train": False,
                    "total_epochs": 1,
                    "profile_steps": None,
                    "use_epsilon": False,
                    "balance_batch": False,
                    "critic_warmup": 0,
                    "rollout_data_dir": None,
                    "test_freq": 1,
                    "save_freq": -1,
                    "save_last": False,
                    "esi_redundant_time": 0,
                },
                "actor_rollout_ref": {
                    "rollout": {"name": "vllm", "n": 2, "multi_turn": {"enable": False}, "max_model_len": 4096},
                    "actor": {"loss_agg_mode": "token-mean"},
                },
                "algorithm": {
                    "adv_estimator": "grpo",
                    "use_kl_in_reward": False,
                    "gamma": 1.0,
                    "lam": 1.0,
                    "norm_adv_by_std_in_grpo": True,
                },
                "reward_model": {"launch_reward_fn_async": False},
            }
        )
        trainer.total_training_steps = 2
        trainer.async_rollout_mode = False
        trainer.use_reference_policy = trainer.use_critic = trainer.use_rm = False
        trainer.reward_fn = Mock()
        trainer.val_reward_fn = Mock()
        trainer.resource_pool_manager = Mock()
        trainer.resource_pool_manager.get_n_gpus.return_value = 1
        trainer._load_checkpoint = Mock()
        trainer._validate = Mock(return_value={"sft_val/accuracy": 0.75, "val/accuracy/hint_0": 0.5})
        trainer._maybe_save_best_checkpoint = Mock()
        trainer._method_d_sft = Mock(return_value={"sft_loss": 0.7, "sft_accuracy": 0.5})
        rollout = rollouts([20, 20])
        rollout.meta_info["timing"] = {}
        trainer.actor_rollout_wg.generate_sequences.side_effect = lambda batch: DataProto.from_dict(
            tensors={key: value.clone() for key, value in rollout.batch.items()},
            meta_info={"timing": {}},
        )
        trainer.actor_rollout_wg.compute_log_prob.return_value = DataProto.from_dict(
            tensors={
                "old_log_probs": torch.zeros_like(rollout.batch["responses"], dtype=torch.float),
                "entropys": torch.zeros_like(rollout.batch["responses"], dtype=torch.float),
            }
        )
        raw = {
            "input_ids": torch.tensor([[0, 4, 5]]),
            "attention_mask": torch.tensor([[0, 1, 1]]),
            "position_ids": torch.tensor([[0, 0, 1]]),
            "raw_prompt_ids": np.array([[4, 5]], dtype=object),
        }
        trainer.train_dataloader = [raw.copy(), raw.copy()]
        rewards = torch.zeros_like(rollout.batch["responses"], dtype=torch.float)
        rewards[:, -1] = torch.tensor([0.1, 1.0])
        tracking = Mock()
        with (
            patch("verl.utils.tracking.Tracking", return_value=tracking),
            patch("verl.trainer.ppo.ray_trainer.compute_reward", return_value=(rewards, {"correct": [False, True]})),
            patch("verl.trainer.ppo.ray_trainer.compute_data_metrics", return_value={"accuracy/overall": 0.5}),
            patch("verl.trainer.ppo.ray_trainer.should_save_ckpt_esi", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            trainer.fit()
        self.assertEqual(trainer.actor_rollout_wg.generate_sequences.call_count, 2)
        trainer._method_d_sft.assert_called_once()
        self.assertEqual(trainer._method_d_sft.call_args.args[1], [False, True])
        trainer.actor_rollout_wg.compute_log_prob.assert_called_once()
        trainer.actor_rollout_wg.update_actor.assert_called_once()
        self.assertEqual(trainer._validate.call_count, 2)
        logs = [call.kwargs["data"] for call in tracking.log.call_args_list]
        self.assertIn("sft_accuracy", logs[0])
        self.assertNotIn("accuracy/overall", logs[0])
        self.assertIn("accuracy/overall", logs[1])
        self.assertTrue(all("sft_val/accuracy" in log and "val/accuracy/hint_0" in log for log in logs))

    def test_validation_weights_prefix_counts_and_retains_answer_accuracy_metrics(self):
        trainer = self.trainer()
        trainer.config = OmegaConf.create(
            {
                "trainer": {"method_d": True, "logger": [], "validation_data_dir": None},
                "actor_rollout_ref": {"rollout": {"val_kwargs": {"n": 1, "do_sample": True}}},
                "reward_model": {"enable": False},
            }
        )
        trainer.async_rollout_mode = False
        trainer.actor_rollout_wg.world_size = 1
        trainer.global_steps = 0
        trainer._maybe_log_val_generations = Mock()
        trainer._method_d_sft = Mock(
            side_effect=[
                {"sft_loss": 1.0, "sft_token_accuracy": 0.0, "sft_accuracy": 0.0, "sft_num_examples": 1,
                 "sft_groups/hint_2/accuracy": 0., "sft_groups/hint_2/token_accuracy": 0.,
                 "sft_groups/hint_2/num_examples": 1,
                 "sft_groups/position_first/accuracy": 0., "sft_groups/position_first/num_examples": 1},
                {"sft_loss": 0.2, "sft_token_accuracy": 1.0, "sft_accuracy": 1.0, "sft_num_examples": 3,
                 "sft_groups/hint_2/accuracy": 1., "sft_groups/hint_2/token_accuracy": 1.,
                 "sft_groups/hint_2/num_examples": 3,
                 "sft_groups/hint_0/accuracy": 1., "sft_groups/hint_0/num_examples": 1,
                 "sft_groups/position_first/accuracy": 1., "sft_groups/position_first/num_examples": 1},
            ]
        )
        trainer.val_dataloader = [
            {
                "input_ids": torch.tensor([[0, 4, 5]]),
                "attention_mask": torch.tensor([[0, 1, 1]]),
                "position_ids": torch.tensor([[0, 0, 1]]),
                "raw_prompt_ids": np.array([[4, 5]], dtype=object),
                "data_source": np.array(["test"], dtype=object),
            }
            for _ in range(2)
        ]
        trainer.actor_rollout_wg.generate_sequences.side_effect = [rollouts([20]), rollouts([300])]
        trainer.val_reward_fn = Mock(
            side_effect=[
                {"reward_tensor": torch.tensor([[0.1]]), "reward_extra_info": {"correct": [False], "num_hints": [0]}},
                {"reward_tensor": torch.tensor([[1.0]]), "reward_extra_info": {"correct": [True], "num_hints": [0]}},
            ]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = trainer._validate()
        self.assertAlmostEqual(metrics["sft_val/loss"], 0.4)
        self.assertEqual(metrics["sft_val/accuracy"], 0.75)
        self.assertEqual(metrics["sft_val/token_accuracy"], 0.75)
        self.assertEqual(metrics["sft_val/num_examples"], 4)
        self.assertEqual(metrics["sft_val/hint_2/accuracy"], 0.75)
        self.assertEqual(metrics["sft_val/hint_2/token_accuracy"], 0.75)
        self.assertEqual(metrics["sft_val/hint_2/num_examples"], 4)
        self.assertEqual(metrics["sft_val/hint_0/accuracy"], 1.)
        self.assertEqual(metrics["sft_val/position_first/accuracy"], 0.5)
        self.assertEqual(metrics["sft_val/position_first/num_examples"], 2)
        self.assertEqual(metrics["val/accuracy/hint_0"], 0.5)
        self.assertTrue(any(key.startswith("val-core/test/reward/") for key in metrics))


def distributed_sft_worker(rank, rendezvous, output_dir):
    from torch.nn.parallel import DistributedDataParallel

    torch.distributed.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.config = OmegaConf.create({"sft_micro_batch_size": 1})
        actor.use_ulysses_sp = actor.use_fused_kernels = False
        actor.device_name = "cpu"
        actor.actor_module = DistributedDataParallel(TinyModel())
        actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.1)

        def step():
            actor.actor_optimizer.step()
            return torch.tensor(0.0)

        actor._optimizer_step = step
        trajectories = rollouts([20, 30])
        trajectories.non_tensor_batch["hint_level"] = np.array([0, 2])
        data = build_prefix_batch(
            trajectories, [1, 1] if rank == 0 else [0, 0], Tokenizer(), report=True,
        )
        if rank == 1:
            data.batch["sft_weights"][-1] = 0
        with (
            patch("verl.workers.actor.dp_actor.get_device_id", return_value="cpu"),
            patch("verl.utils.debug.performance._get_current_mem_info", return_value=("0",) * 4),
        ):
            actor.method_d_sft(data, update=True)
            metrics = actor.method_d_sft(data, update=False)
        torch.save(
            {"logits": actor.actor_module.module.logits.detach(), "metrics": metrics}, Path(output_dir) / f"{rank}.pt"
        )
    finally:
        torch.distributed.destroy_process_group()


class MethodDDistributedTests(unittest.TestCase):
    def test_global_weighting_matches_single_batch_with_uneven_real_prefix_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                distributed_sft_worker, args=(str(Path(directory) / "rdzv"), directory), nprocs=2, join=True
            )
            ranks = [torch.load(Path(directory) / f"{rank}.pt", weights_only=True) for rank in range(2)]
        expected = torch.full((10,), -0.01)
        expected[2] += 0.1 / 3
        expected[3] += 0.2 / 3
        for result in ranks:
            torch.testing.assert_close(result["logits"], expected)
            self.assertEqual(result["metrics"]["sft_num_examples"], 3)
            self.assertAlmostEqual(result["metrics"]["sft_accuracy"], 2 / 3, places=6)
            self.assertEqual(result["metrics"]["sft_groups/hint_0/num_examples"], 2)
            self.assertEqual(result["metrics"]["sft_groups/hint_2/num_examples"], 1)
            self.assertEqual(result["metrics"]["sft_groups/position_first/num_examples"], 3)
            self.assertEqual(result["metrics"]["sft_groups/position_final/num_examples"], 3)
            self.assertEqual(result["metrics"]["sft_groups/hint_0/accuracy"], 0.5)
            self.assertEqual(result["metrics"]["sft_groups/hint_2/accuracy"], 1.)


if __name__ == "__main__":
    unittest.main()
