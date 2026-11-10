import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from omegaconf import OmegaConf

from test_method_d import Tokenizer, rollouts
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.method_e import (
    attach_classification_head, build_trajectory_batch, classification_statistics,
)


def small_model():
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    ))
    attach_classification_head(model)
    return model


def fsdp_worker(rank, rendezvous, output_dir):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
    )
    try:
        for head_only in (False, True):
            model = small_model()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
            actor.config = OmegaConf.create({"sft_micro_batch_size": 2, "sft_head_only": head_only})
            actor.device_name = "cuda"
            actor.actor_module = FSDP(model, device_id=rank, use_orig_params=True, sync_module_states=True)
            actor.actor_optimizer = torch.optim.AdamW(actor.actor_module.parameters(), lr=0.01)
            def step():
                norm = actor.actor_module.clip_grad_norm_(1.)
                actor.actor_optimizer.step()
                return norm
            actor._optimizer_step = step
            data = build_trajectory_batch(
                rollouts([3, 5]), [rank, rank], Tokenizer(), stride=2, report=True,
            )
            if rank == 1:
                data.batch["sft_weights"][-1] = 0
            # Seed optimizer state through the LM path, as on an interleaved RL step.
            actor.actor_module(
                input_ids=data.batch["input_ids"].cuda(),
                attention_mask=data.batch["attention_mask"].cuda(),
            ).logits.float().mean().backward()
            actor.actor_optimizer.step()
            actor.actor_optimizer.zero_grad()
            before = {name: p.detach().clone() for name, p in actor.actor_module.named_parameters()}
            with patch("verl.workers.actor.dp_actor.get_device_id", return_value=rank):
                result = actor.method_e_sft(data, update=True)
                evaluated = actor.method_e_sft(data, update=False)
            if head_only:
                for name, p in actor.actor_module.named_parameters():
                    if "classification_head." not in name:
                        torch.testing.assert_close(before[name], p, atol=0, rtol=0)
            assert result["sft_num_examples"] == 7
            assert evaluated["sft_num_examples"] == 7
            assert evaluated["sft_groups/hint_0/num_examples"] == 7
            assert evaluated["sft_groups/position_first/num_examples"] == 3
            assert evaluated["sft_groups/position_final/num_examples"] == 3
            assert torch.isfinite(torch.tensor(result["sft_loss"]))
            # The ordinary LM forward still works after a classification update.
            assert actor.actor_module(input_ids=torch.tensor([[4, 5]], device=rank)).logits.shape[-1] == 32
        torch.save(result, Path(output_dir) / f"{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


class MethodETests(unittest.TestCase):
    def test_trainer_routes_whole_trajectories_and_masks_distributed_padding(self):
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.config = OmegaConf.create({
            "trainer": {"method_e": True, "sft_stride": 2},
            "actor_rollout_ref": {"rollout": {"max_model_len": 4096}},
        })
        trainer.tokenizer = Tokenizer()
        trainer.actor_rollout_wg = Mock(world_size=4)
        trainer.actor_rollout_wg.update_actor.return_value = DataProto(
            meta_info={"metrics": {"sft_accuracy": [1.]}}
        )
        trainer._method_d_sft(rollouts([3, 5]), [0, 1], update=True)
        batch = trainer.actor_rollout_wg.update_actor.call_args.args[0]
        self.assertEqual(len(batch), 4)
        self.assertEqual(batch.batch["sft_weights"].sum(-1).tolist(), [2, 3, 0, 0])
        self.assertTrue(batch.meta_info["method_e_sft"])
        self.assertNotIn("sft_label_ids", batch.meta_info)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two GPUs for FSDP")
    def test_two_gpu_fsdp_with_interleaved_lm_and_head_only_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                fsdp_worker, args=(str(Path(directory) / "rdzv"), directory),
                nprocs=2, join=True,
            )
            results = [torch.load(Path(directory) / f"{rank}.pt", weights_only=True) for rank in range(2)]
            self.assertEqual(results[0]["sft_loss"], results[1]["sft_loss"])

    def test_one_sequence_per_rollout_with_exact_causal_boundary_positions(self):
        batch = build_trajectory_batch(rollouts([300, 256, 20, 0]), [0, 1, 0, 1], Tokenizer())
        self.assertEqual(len(batch), 4)
        self.assertEqual(batch.batch["sft_weights"].sum(-1).tolist(), [3, 2, 1, 1])
        for i, ends in enumerate(([128, 256, 300], [128, 256], [20], [0])):
            offset = int((batch.batch["attention_mask"][i] == 0).sum())
            positions = (batch.batch["sft_weights"][i] > 0).nonzero().flatten().tolist()
            self.assertEqual(positions, [offset + 2 + end - 1 for end in ends])
            self.assertEqual(batch.batch["sft_targets"][i, positions].tolist(),
                             [[0, 1, 0, 1][i]] * len(ends))
        self.assertFalse((batch.batch["input_ids"] == 9).any())

    def test_full_sequence_matches_prefix_and_bypasses_lm_head(self):
        model = small_model().eval()
        ids = torch.tensor([[4, 5, 8, 8, 8]])
        calls = []
        handle = model.lm_head.register_forward_hook(lambda *args: calls.append(True))
        logits = model(input_ids=ids, method_e=True, use_cache=False)
        prefix = model(input_ids=ids[:, :3], method_e=True, use_cache=False)
        torch.testing.assert_close(logits[:, 2], prefix[:, -1])
        self.assertEqual(calls, [])
        self.assertEqual(logits.shape, (1, 5, 2))
        self.assertEqual(model(input_ids=ids).logits.shape, (1, 5, 32))
        self.assertEqual(calls, [True])
        handle.remove()

    def test_loss_only_at_selected_positions_and_both_training_options(self):
        for head_only in (False, True):
            with self.subTest(head_only=head_only):
                model = small_model()
                batch = build_trajectory_batch(rollouts([3, 2]), [0, 1], Tokenizer(), stride=2).batch
                logits = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    position_ids=batch["position_ids"], method_e=True, head_only=head_only,
                )
                stats = classification_statistics(logits, batch["sft_targets"], batch["sft_weights"])
                self.assertEqual(stats["num_examples"].item(), 3)
                stats["loss"].backward()
                self.assertIsNotNone(model.classification_head.weight.grad)
                self.assertEqual(model.model.embed_tokens.weight.grad is None, head_only)
                self.assertIsNone(model.lm_head.weight.grad)

    def test_checkpoint_roundtrip_preserves_head_and_lm(self):
        model = small_model().eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = Qwen2ForCausalLM.from_pretrained(directory)
            attach_classification_head(loaded, directory)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[name])

    def test_actor_batches_trajectories_and_head_only_does_not_decay_backbone(self):
        for head_only in (False, True):
            with self.subTest(head_only=head_only):
                actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
                actor.config = OmegaConf.create({"sft_micro_batch_size": 2, "sft_head_only": head_only})
                actor.device_name = "cpu"
                actor.actor_module = small_model()
                actor.actor_optimizer = torch.optim.AdamW(
                    actor.actor_module.parameters(), lr=0.01, weight_decay=0.1,
                )
                before = {name: p.detach().clone() for name, p in actor.actor_module.named_parameters()}
                def step():
                    actor.actor_optimizer.step()
                    return torch.tensor(0.)
                actor._optimizer_step = step
                data = build_trajectory_batch(rollouts([3, 5, 2]), [0, 1, 0], Tokenizer(), stride=2)
                data.batch["sft_weights"][-1] = 0
                with (
                    patch("verl.workers.actor.dp_actor.get_device_id", return_value="cpu"),
                    patch("torch.distributed.all_reduce"),
                    patch("torch.distributed.get_world_size", return_value=1),
                    patch.object(actor.actor_module, "forward", wraps=actor.actor_module.forward) as forward,
                ):
                    metrics = actor.method_e_sft(data, update=True)
                    self.assertEqual(forward.call_count, 2)
                self.assertEqual(metrics["sft_num_examples"], 5)
                self.assertNotIn("sft_token_accuracy", metrics)
                self.assertFalse(torch.equal(
                    before["classification_head.weight"], actor.actor_module.classification_head.weight,
                ))
                if head_only:
                    for name, parameter in actor.actor_module.named_parameters():
                        if not name.startswith("classification_head."):
                            torch.testing.assert_close(before[name], parameter, rtol=0, atol=0)

    def test_invalid_inputs(self):
        for labels, stride in (([], 128), ([2], 128), ([0], 0)):
            with self.assertRaises(ValueError):
                build_trajectory_batch(rollouts([3]), labels, Tokenizer(), stride)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_trajectory_batch(rollouts([3]), [0], Tokenizer(), max_model_len=3)


if __name__ == "__main__":
    unittest.main()
