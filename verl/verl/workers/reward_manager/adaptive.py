"""Adaptive reward manager with EMA-based dynamic abstention reward.

Maintains a running EMA of the mean attempt reward across training steps.
The abstention reward r_a is set to:

    r_a(t+1) = clamp(R_att_ema(t) - delta, r_min, r_max)

This self-calibrates the abstention threshold to the model's current ability,
avoiding collapse to always-abstain or never-abstain.

See DYNAMIC_ABSTENTION.md for the full specification.
"""

from collections import defaultdict

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.batch import BatchRewardManager


@register("adaptive")
class AdaptiveRewardManager(BatchRewardManager):
    """Reward manager with EMA-tracked adaptive abstention reward.

    Extends BatchRewardManager with cross-step state: after each batch,
    computes the problem-level mean attempt reward, updates an EMA, and
    derives the abstention reward for the next step.

    Additional kwargs (passed via reward_kwargs in method config):
        beta: EMA decay factor (default: 0.95)
        delta: Selectivity margin (default: 0.05)
        r_min: Floor clamp for r_a (default: 0.15)
        r_max: Ceiling clamp for r_a (default: 0.9)
        r_c: Reward for correct answer (default: 1.0)
        r_w: Reward for wrong answer (default: 0.1)
    """

    def __init__(self, tokenizer, num_examine=0, compute_score=None,
                 reward_fn_key="data_source", **reward_kwargs):
        # Extract EMA hyperparameters before forwarding to parent
        self.beta = float(reward_kwargs.pop("beta", 0.95))
        self.delta = float(reward_kwargs.pop("delta", 0.05))
        self.r_min = float(reward_kwargs.pop("r_min", 0.15))
        self.r_max = float(reward_kwargs.pop("r_max", 0.9))
        # Also extract r_c/r_w for use in __call__, but don't forward to parent
        # (the wrapped compute_score already has these baked in via get_custom_reward_fn)
        self._r_c = float(reward_kwargs.pop("r_c", 1.0))
        self._r_w = float(reward_kwargs.pop("r_w", 0.1))

        super().__init__(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=reward_fn_key,
            **reward_kwargs,
        )

        # EMA state — initialized on first batch
        self.r_att_ema = None  # Will be set after first batch
        self.r_a = None  # Current abstention reward (None = use first-batch init)
        self._initialized = False

    def __call__(self, data: DataProto, return_dict=False):
        """Compute rewards with adaptive abstention, then update EMA state."""
        # If there are pre-computed rm scores, defer to parent
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # --- Step 1: Get per-sample classifications from the reward function ---
        scores = self.verify(data)

        # --- Step 2: Compute adaptive r_a and assign rewards ---
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        uids = data.non_tensor_batch.get("uid", None)

        r_c = self._r_c
        r_w = self._r_w

        # Group by uid to compute problem-level statistics
        if uids is not None:
            groups = defaultdict(list)
            for i, uid in enumerate(uids):
                groups[uid].append(i)
        else:
            # Fallback: treat entire batch as one group
            groups = {"all": list(range(len(scores)))}

        # Compute per-problem reward (two-level average, all rollouts included).
        # Abstained rollouts are scored as r_w for the EMA calculation, so that
        # heavy abstention drags the EMA down and self-corrects.
        problem_all_rewards = []  # For EMA: includes abstained as r_w
        problem_attempt_rewards = []  # For logging: attempted-only average
        total_attempted = 0
        total_abstained = 0
        total_correct_attempted = 0
        total_attempts = 0

        for uid, indices in groups.items():
            all_rewards = []  # All rollouts, abstained scored as r_w
            attempted_rewards = []  # Attempted rollouts only
            n_a = 0
            for i in indices:
                score = scores[i]
                if isinstance(score, dict) and score.get("abstained", False):
                    n_a += 1
                    all_rewards.append(r_w)  # Count abstained as r_w for EMA
                else:
                    if isinstance(score, dict):
                        r = score["score"]
                        if score.get("correct", False):
                            total_correct_attempted += 1
                    else:
                        r = score
                    attempted_rewards.append(r)
                    all_rewards.append(r)

            total_abstained += n_a
            total_attempted += len(attempted_rewards)
            total_attempts += len(indices)

            # EMA input: all rollouts (abstained counted as r_w)
            problem_all_rewards.append(sum(all_rewards) / len(all_rewards))
            # Logging: attempted-only average
            if attempted_rewards:
                problem_attempt_rewards.append(sum(attempted_rewards) / len(attempted_rewards))

        # --- Step 3: Initialize or update EMA ---
        just_initialized = False
        # EMA uses all-rollout average (abstained = r_w)
        r_all_batch = sum(problem_all_rewards) / len(problem_all_rewards) if problem_all_rewards else None
        # Attempted-only average for logging
        r_att_batch = sum(problem_attempt_rewards) / len(problem_attempt_rewards) if problem_attempt_rewards else None

        if r_all_batch is not None:
            if not self._initialized:
                # First batch: initialize EMA directly
                self.r_att_ema = r_all_batch
                self.r_a = max(self.r_min, min(self.r_max, r_all_batch - self.delta))
                self._initialized = True
                just_initialized = True
            # else: EMA was already updated at end of previous call; r_a is current

            r_att_batch_flat = (
                sum(r for s in scores for r in [s["score"] if isinstance(s, dict) else s]
                    if not (isinstance(s, dict) and s.get("abstained", False)))
                / max(total_attempted, 1)
            ) if total_attempted > 0 else None
        else:
            r_att_batch_flat = None

        # Use current r_a (or fallback for edge case)
        current_r_a = self.r_a if self.r_a is not None else 0.5

        # --- Step 4: Assign rewards using current r_a ---
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        rewards = []
        already_printed = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                if score.get("correct", False):
                    reward = r_c
                elif score.get("abstained", False):
                    reward = current_r_a
                else:
                    reward = score["score"]  # format_score or 0

                # Propagate all original keys plus adaptive-specific ones
                for key, value in score.items():
                    if key != "score":
                        reward_extra_info[key].append(value)
                reward_extra_info["score"].append(reward)
            else:
                reward = score

            # Adaptive-specific metrics per sample
            reward_extra_info["r_a"].append(current_r_a)

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(
                    data.batch["responses"][i][:length], skip_special_tokens=True
                )
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", reward)
                print("[r_a]", current_r_a)
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        # --- Step 5: Post-step EMA update for NEXT step ---
        # EMA uses r_all_batch (abstained counted as r_w) to create negative feedback:
        # high abstention → EMA drops → r_a drops → less abstention
        # Skip if just initialized (first batch already set EMA directly)
        if r_all_batch is not None and self._initialized and not just_initialized:
            self.r_att_ema = self.beta * self.r_att_ema + (1 - self.beta) * r_all_batch
            self.r_a = max(self.r_min, min(self.r_max, self.r_att_ema - self.delta))

        # Add batch-level metrics to reward_extra_info for logging
        abstention_rate = total_abstained / max(total_attempts, 1)
        n_problems = len(groups)
        n_fully_abstained = sum(
            1 for uid, indices in groups.items()
            if all(
                isinstance(scores[i], dict) and scores[i].get("abstained", False)
                for i in indices
            )
        )
        attempted_accuracy = total_correct_attempted / max(total_attempted, 1)
        threshold_p = (current_r_a - r_w) / max(r_c - r_w, 1e-8)

        # Broadcast batch-level metrics to all samples for logging
        r_all_str = f"{r_all_batch:.4f}" if r_all_batch is not None else "N/A"
        r_att_str = f"{r_att_batch:.4f}" if r_att_batch is not None else "N/A"
        for _ in range(len(data)):
            reward_extra_info["r_a_adaptive"].append(current_r_a)
            reward_extra_info["R_att_ema"].append(self.r_att_ema if self.r_att_ema is not None else 0.0)
            if r_all_batch is not None:
                reward_extra_info["R_all_batch"].append(r_all_batch)
            if r_att_batch is not None:
                reward_extra_info["R_att_batch"].append(r_att_batch)
            if r_att_batch_flat is not None:
                reward_extra_info["R_att_batch_flat"].append(r_att_batch_flat)
            reward_extra_info["abstention_rate"].append(abstention_rate)
            reward_extra_info["abstention_rate_by_problem"].append(n_fully_abstained / max(n_problems, 1))
            reward_extra_info["abstention_threshold_p"].append(threshold_p)
            reward_extra_info["attempted_accuracy"].append(attempted_accuracy)

        ema_str = f"{self.r_att_ema:.4f}" if self.r_att_ema is not None else "N/A"
        print(
            f"[AdaptiveRewardManager] r_a={current_r_a:.4f}, "
            f"R_ema={ema_str}, "
            f"R_all_batch={r_all_str}, "
            f"R_att_batch={r_att_str}, "
            f"abs_rate={abstention_rate:.3f}, "
            f"att_acc={attempted_accuracy:.3f}, "
            f"thresh_p={threshold_p:.3f}"
        )

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(reward_extra_info)}
        else:
            return reward_tensor

    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {
            "r_att_ema": self.r_att_ema,
            "r_a": self.r_a,
            "_initialized": self._initialized,
        }

    def load_state_dict(self, state):
        """Restore EMA state from checkpoint."""
        self.r_att_ema = state["r_att_ema"]
        self.r_a = state["r_a"]
        self._initialized = state["_initialized"]
