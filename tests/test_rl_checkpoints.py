import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

REWARD_KEY = "val-core/math/reward/mean@1"
HINT_KEY = "val/accuracy/hint_0"


class RlCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.trainer = self.new_trainer()

    def new_trainer(self, multi=True):
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.config = OmegaConf.create(
            {
                "trainer": {
                    "default_local_dir": str(self.root),
                    "default_hdfs_dir": None,
                    "save_best": True,
                    "save_hint_bests": multi,
                    "max_actor_ckpt_to_keep": 2,
                    "max_critic_ckpt_to_keep": 2,
                }
            }
        )
        trainer.actor_rollout_wg = Mock()
        trainer.actor_rollout_wg.save_checkpoint.side_effect = self.save_worker
        trainer.critic_wg = Mock()
        trainer.critic_wg.save_checkpoint.side_effect = self.save_worker
        trainer.train_dataloader = Mock()
        trainer.train_dataloader.state_dict.return_value = {"position": 0}
        trainer.reward_fn = object()
        trainer.use_critic = False
        return trainer

    @staticmethod
    def save_worker(local_path, remote_path, step, max_ckpt_to_keep):
        path = Path(local_path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "weights.pt").write_text(f"step={step}")

    def validate(self, step, reward, hint_accuracy, trainer=None):
        trainer = trainer or self.trainer
        trainer.global_steps = step
        with contextlib.redirect_stdout(io.StringIO()):
            trainer._maybe_save_best_checkpoint({REWARD_KEY: reward, HINT_KEY: hint_accuracy})

    def remaining(self):
        return {int(path.name.rsplit("_", 1)[-1]) for path in self.root.glob("global_step_*")}

    def records(self):
        return json.loads((self.root / "best_checkpoints.json").read_text())

    def test_all_three_criteria_share_one_save(self):
        self.validate(25, 0.5, 0.5)
        self.assertEqual(self.remaining(), {25})
        self.assertEqual(set(self.records()), {"best", "best_hint_0", "best_combined"})
        self.assertEqual({record["step"] for record in self.records().values()}, {25})
        self.trainer.actor_rollout_wg.save_checkpoint.assert_called_once_with(
            str(self.root / "global_step_25" / "actor"), None, 25, max_ckpt_to_keep=None
        )
        legacy = json.loads((self.root / "best_checkpoint.json").read_text())
        self.assertEqual(legacy, self.records()["best"])

    def test_distinct_winners_are_protected_even_when_old(self):
        self.validate(25, 0.9, 0.1)
        self.validate(50, 0.2, 0.8)
        self.validate(75, 0.7, 0.6)
        self.validate(100, 0.6, 0.6)
        self.assertEqual(self.remaining(), {25, 50, 75})
        self.assertEqual(self.records()["best"]["step"], 25)
        self.assertEqual(self.records()["best_hint_0"]["step"], 50)
        self.assertEqual(self.records()["best_combined"]["step"], 75)
        self.assertAlmostEqual(self.records()["best_combined"]["score"], 0.65)

        self.trainer.global_steps = 125
        with contextlib.redirect_stdout(io.StringIO()):
            self.trainer._save_checkpoint()
        self.assertEqual(self.remaining(), {25, 50, 75, 125})
        self.assertEqual((self.root / "latest_checkpointed_iteration.txt").read_text(), "125")
        self.trainer.global_steps = 150
        with contextlib.redirect_stdout(io.StringIO()):
            self.trainer._save_checkpoint()
        self.assertEqual(self.remaining(), {25, 50, 75, 150})

    def test_replacing_winners_removes_superseded_steps(self):
        self.validate(25, 0.9, 0.1)
        self.validate(50, 0.2, 0.8)
        self.validate(75, 0.7, 0.6)
        self.validate(100, 1.0, 1.0)
        self.assertEqual(self.remaining(), {100})
        self.assertEqual({record["step"] for record in self.records().values()}, {100})

    def test_ties_keep_earlier_winner_and_dont_save(self):
        self.validate(25, 0.5, 0.5)
        self.validate(50, 0.5, 0.5)
        self.assertEqual(self.remaining(), {25})
        self.assertEqual(self.trainer.actor_rollout_wg.save_checkpoint.call_count, 1)

    def test_resume_preserves_each_previous_best(self):
        self.validate(25, 0.9, 0.1)
        self.validate(50, 0.2, 0.8)
        self.validate(75, 0.7, 0.6)
        resumed = self.new_trainer()
        self.validate(100, 0.4, 0.4, resumed)
        resumed.actor_rollout_wg.save_checkpoint.assert_not_called()
        self.validate(125, 0.95, 0.1, resumed)
        self.assertEqual(self.remaining(), {50, 75, 125})
        self.assertEqual(self.records()["best_hint_0"]["step"], 50)
        self.assertEqual(self.records()["best_combined"]["step"], 75)

    def test_legacy_best_record_is_preserved_when_enabling_multi_metric(self):
        legacy = self.new_trainer(multi=False)
        self.validate(25, 0.9, 0.1, legacy)
        self.validate(50, 0.2, 0.5)
        self.assertEqual(self.records()["best"]["step"], 25)
        self.assertEqual(self.records()["best_hint_0"]["step"], 50)
        self.assertEqual(self.remaining(), {25, 50})

    def test_single_metric_mode_keeps_existing_worker_limit(self):
        trainer = self.new_trainer(multi=False)
        self.validate(25, 0.9, 0.1, trainer)
        trainer.actor_rollout_wg.save_checkpoint.assert_called_once_with(
            str(self.root / "global_step_25" / "actor"), None, 25, max_ckpt_to_keep=2
        )
        self.assertFalse((self.root / "best_checkpoints.json").exists())

    def test_periodic_and_final_save_dont_duplicate_same_step(self):
        self.validate(25, 0.5, 0.5)
        with contextlib.redirect_stdout(io.StringIO()):
            self.trainer._save_checkpoint()
            self.trainer._save_checkpoint()
        self.assertEqual(self.trainer.actor_rollout_wg.save_checkpoint.call_count, 1)

    def test_critic_worker_does_not_prune_metric_winners(self):
        self.trainer.use_critic = True
        self.validate(25, 0.5, 0.5)
        self.trainer.critic_wg.save_checkpoint.assert_called_once_with(
            str(self.root / "global_step_25" / "critic"), None, 25, max_ckpt_to_keep=None
        )

    def test_missing_metric_fails_before_saving(self):
        self.trainer.global_steps = 25
        with self.assertRaisesRegex(KeyError, HINT_KEY):
            self.trainer._maybe_save_best_checkpoint({REWARD_KEY: 0.5})
        self.trainer.actor_rollout_wg.save_checkpoint.assert_not_called()

    def test_nonfinite_metric_fails_before_saving(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.validate(25, 0.5, float("nan"))
        self.trainer.actor_rollout_wg.save_checkpoint.assert_not_called()

    def test_missing_selected_checkpoint_stops_pruning(self):
        self.validate(25, 0.9, 0.1)
        (self.root / "global_step_25" / "actor").rename(self.root / "saved_actor")
        with self.assertRaisesRegex(FileNotFoundError, "global_step_25"):
            self.validate(50, 0.2, 0.8)
        self.assertTrue((self.root / "global_step_50" / "actor").is_dir())

    def test_changing_selection_metric_on_resume_fails(self):
        self.validate(25, 0.5, 0.5)
        self.trainer.config.trainer.best_ckpt_metric = HINT_KEY
        with self.assertRaisesRegex(ValueError, "Checkpoint metric for best changed"):
            self.validate(50, 0.8, 0.8)

    def test_failed_save_does_not_advance_best_records(self):
        self.validate(25, 0.5, 0.5)
        records = self.records()
        self.trainer.actor_rollout_wg.save_checkpoint.side_effect = OSError("disk full")
        with self.assertRaisesRegex(OSError, "disk full"):
            self.validate(50, 1.0, 1.0)
        self.assertEqual(self.records(), records)


if __name__ == "__main__":
    unittest.main()
