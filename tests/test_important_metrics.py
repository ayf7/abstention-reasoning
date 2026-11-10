import unittest
from unittest.mock import Mock

from verl.utils.tracking import IMPORTANT_METRICS, Tracking


class ImportantMetricsTests(unittest.TestCase):
    def test_only_selected_metrics_are_mirrored_to_wandb(self):
        tracking = Tracking.__new__(Tracking)
        wandb = Mock()
        console = Mock()
        tracking.logger = {"wandb": wandb, "console": console}
        data = dict.fromkeys(IMPORTANT_METRICS, 0.5)
        data.update({"critic/score/max": 1.0, "precision/hints_3": 0.2})
        original = data.copy()

        tracking.log(data, step=25)

        expected = {**data, **{f"important/{key}": 0.5 for key in IMPORTANT_METRICS}}
        wandb.log.assert_called_once_with(data=expected, step=25)
        console.log.assert_called_once_with(data=original, step=25)
        self.assertEqual(data, original)
        self.assertEqual(len(IMPORTANT_METRICS), 14)

    def test_backend_filter_and_missing_metrics_are_preserved(self):
        tracking = Tracking.__new__(Tracking)
        wandb = Mock()
        console = Mock()
        tracking.logger = {"wandb": wandb, "console": console}

        tracking.log({"val/accuracy/hint_0": 0.25}, step=0, backend=["wandb"])

        wandb.log.assert_called_once_with(
            data={"val/accuracy/hint_0": 0.25, "important/val/accuracy/hint_0": 0.25}, step=0
        )
        console.log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
