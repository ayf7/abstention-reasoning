# Pending patches

## turn_cap_forced_answer.patch — APPLIED on this branch (new_hint_setting)

A rollout that spends its last `max_turns` turn receiving a hint exits
`agentic_loop` still incomplete and mid-`<think>`. `idx_to_truncation` is never
set for turn-cap exhaustion, so `_force_answers` skips it and the reward
function scores it 0 as malformed.

Measured on `qwen2.5-3b-math-hint-p0` (job 745395, steps 68-70): malformed rate
by hint count was <=5.6% for 0-5 hints and **100% (53/53) at 6 hints**. The
penalty therefore lands only on the heaviest hint users and grows with hint
use, which is the same structural bias documented in `logs/diagnosis.md`.

Verified: the existing `_answer_scaffold` already handles this state (six
complete hint cycles ending mid-`<think>`), returning `"\n</think>\n\n<answer>"`.
Replaying all 53 real failures through scaffold + the real
`has_malformed_structure` gives 53/53 valid, 0 declined, 0 still malformed.

Applied here; still parked on `rl_runs_2`, deliberately. Jobs 745395 (p=0, p=0.1) and
763768 (p=0.05) were launched without it and read source from this tree at
process start; `genai-goyal-highpri` is `PreemptMode=REQUEUE`, so a requeue
would restart an arm on patched code and put a code difference inside the
ablation.
