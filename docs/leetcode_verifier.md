# LeetCode task: verifier design

Status: infrastructure phase, LeetCodeDataset only (2,641 train problems). Goal is a ready pipeline for RL and evaluation, so that a larger dataset can be added later as a data loader only.

## Summary

One verifier service is shared by RL training and `pipeline evaluate`. It runs a fixed pool of worker processes, each pinned to its own physical CPU core. For each submission a worker forks a locked-down child that runs the code on the test inputs and streams back the outputs. The service compares outputs against expected values that the sandboxed code never sees. A submission passes only if every test is correct and its CPU time is within a limit derived from the gold solution.

## Decisions

| Question | Decision |
|---|---|
| Reward | Pass/fail. Pass = all tests correct AND within the timing gate. |
| Isolation | Workers pinned one per physical core (sibling hyperthread left idle). |
| Containment | No root, Docker, or user namespaces on this cluster. Use rlimits + Landlock (filesystem) + seccomp (syscalls) on a forked child. |
| Timing measure | Child CPU time from `wait4` rusage, measured by the worker. The submission cannot forge it. Hardware instruction counters are blocked (`perf_event_paranoid=4`). |
| Dataset | LeetCodeDataset (newfacade) for now; contamination is not a concern (base models). |
| Model | Qwen3-4B-Base. |

## Components

```
                ┌────────── Verifier service (fresh process, never forked from Ray/vLLM) ──────────┐
RL reward  ──┐  │ gold-time cache: spec hash → t_gold (measured lazily on first sight)          │
             ├─►│ dispatcher ──► worker 0 (pinned) ──fork──► sandboxed child                    │
evaluate   ──┘  │            ──► worker 1 (pinned) ──fork──► sandboxed child                    │
  batch of      │            ──► … one worker per physical core                                │
  (spec, code)  └───────────────────────────────────────────────────────────────────────────────┘
                   ◄── one verdict per submission
```

- **Stateless requests.** Each request carries the problem specs it needs (setup, entry point, test inputs, expected outputs, gold code), keyed by a SHA-1 of the spec. The spec lives in each RL row's `ground_truth`, so the service needs no data files and cannot drift from the parquet it scores. Only gold timings are cached, by spec hash.
- **Dispatcher.** The service process. Sends jobs to workers, compares outputs, applies the gate. Clients reach it over a Unix socket (`$VERIFIER_SOCKET`, created mode 0600); if the variable is unset, the client launches a private service on the caller's own physical cores.
- **Workers.** Started with `spawn` (clean interpreters), pinned with `sched_setaffinity`. A worker that dies or hangs is restarted and the job returns `INFRA_ERROR`. Workers receive only code and test inputs. Expected outputs never enter a worker's memory, so the forked child cannot find them via `gc`, frame inspection, or raw memory reads.

## One submission

1. **Extract code**: the contents of `<answer>`, as plain code; if it holds a fenced block, the last block. Missing or empty → `MALFORMED`, nothing runs.
2. **Fork** (≈1 ms vs 12–23 ms for a fresh interpreter; full confinement ≈1.7 ms, warm per-submission overhead 4–10 ms). The child immediately:
   - closes every inherited fd listed in `/proc/self/fd` except its result pipe (a blanket `closerange` cost 7 ms); stdin/stdout/stderr → `/dev/null`;
   - `setsid` (so the worker can kill the whole group);
   - rlimits: address space, CPU seconds, file size 0, open files, no core dumps; stack raised for deep recursion;
   - Landlock: read-only access to the Python installation, nothing else (blocks reading the dataset or repo);
   - seccomp: `EPERM` for exec, fork/clone (threads included), sockets, ptrace, kill, namespace and mount calls.
3. **Run**: the worker pre-imports the modules preambles use (including `sortedcontainers`, needed by 90 problems) before forking. Exec the dataset preamble (`from typing import *`, `ListNode`, `TreeNode`, …) plus the submission. For each test, build a fresh `Solution()` and call the entry method with that test's arguments. Write each result as one canonical JSON line (linked lists and trees become tagged value lists). On an exception, write its type and stop.
4. **Watchdog** (worker): wall-clock deadline, then `killpg(SIGKILL)` and `wait4`. `RLIMIT_CPU` is the backstop. Output size is capped.
5. **Verdict** (dispatcher): compare outputs in order. Python `==` semantics with float tolerance (`1e-5` relative/absolute), lists and tuples treated alike.

| Verdict | Meaning |
|---|---|
| `PASS` | all outputs match and CPU time within the gate |
| `WRONG_ANSWER` | first mismatching test recorded |
| `RUNTIME_ERROR` | exception, crash, or early exit before all outputs |
| `TIMEOUT` | killed at the wall deadline |
| `TOO_SLOW` | correct, but failed the timing gate after re-check |
| `MALFORMED` | no `<answer>`, or an empty one |
| `INFRA_ERROR` | worker or harness failure; logged separately, never silently scored as a model failure |

## Timing gate

- `limit = max(k · t_gold, floor)`; defaults `k = 3`, `floor` set from measured gold times (see Results).
- `t_gold` is measured lazily by the service on its own workers, the first time a problem appears (median of 3 runs), then cached. If gold itself fails at that point, the gate is disabled for that problem (`gold_error` in the result) rather than failing every submission.
- Submission wall deadline: `min(10 s, max(4 · limit, 1 s))`. This always matches the hardware the service runs on; there is no calibration file to go stale.
- **Borderline re-check**: if a correct submission exceeds the limit, the same worker re-runs gold and the submission back to back and decides on that pair. This cancels clock-speed drift.
- With LeetCodeDataset's tiny inputs (median largest input 174 characters), the gate only catches runaway code. Catching slow algorithms needs stress-input generators, which come with dataset augmentation.

## Data

- Tests come from the dataset's `check(candidate)` function, parsed with `ast`: each `assert` gives the call arguments (source) and the expected value. `input_output` is not used because it is lossy (it drops `list_node(...)`/`tree_node(...)` wrappers).
- Expected values are evaluated once at data-build time in the trusted process and stored canonically.
- Every gold solution is run once through the real sandbox at build time. Problems whose gold fails, or whose asserts use an unsupported form, are dropped and counted. This includes golds whose full test suite exceeds the 10 s CPU cap (e.g. bitmask DPs run on ~100 tests), and golds slower than `timeout_s / k` ≈ 3.3 s, whose 3× limit the 10 s wall deadline would cut short.
- Splits: `sft_train`/`sft_val`/`rl_train`/`rl_val` partition the dataset's train rows (0–27 / 27–30 / 30–95 / 95–100%); `eval` is the dataset's own test split, so evaluation never sees train problems.
- Problems whose description says "in any order" are dropped for now: the dataset's tests compare exactly against one ordering, so correct answers in another order would be marked wrong.

## Deployment

**RL.** `verl/recipe/leetcode/reward_function.py` provides a batch scorer for verl's `batch` reward manager, which sends a whole step's responses in one call to the service. `train_rl` starts the service on cores carved from the job's own allocation and pins verl to the rest (Mode A). The verifier takes the last `$VERIFIER_WORKERS` (default 8) physical cores; their hyperthread siblings are left idle. Mode B (service on a separate CPU job, `python -m pipeline.tasks.leetcode.verifier --socket PATH`, reached via `VERIFIER_SOCKET`) uses the same code.

**Evaluation.** `BaseTask.check_correctness_batch` (default: loop over `check_correctness`, so other tasks are unchanged). The LeetCode task overrides it to send the batch to the service. Results record verdict and CPU time per sample.

## Validation checklist

| Check | Pass criterion |
|---|---|
| Gold through harness | failures dropped and counted |
| Gold vs gold timing | ~0 false `TOO_SLOW` at chosen k / floor |
| Adversarial suite (fork bomb, infinite loop, memory bomb, file read/write, sockets, `sys.exit`, `os._exit`, monkeypatching, huge output, signal-catching loops) | contained, correct verdict, no leftover processes |
| Throughput | wall time per 1,024-submission batch at W = 8, 16 |

## Results

### Data build (full LeetCodeDataset, 32-CPU job)

| | Count |
|---|---|
| Kept | 2,701 (train 2,481, test 220) |
| Dropped: "in any order" | 115 |
| Dropped: gold exceeds 10 s CPU | 25 |
| Dropped: gold slower than `timeout_s / k` ≈ 3.3 s | 27 |
| Dropped: gold runtime error | 1 |

Splits: sft_train 669, sft_val 75, rl_train 1,612, rl_val 125, eval 220.

The one runtime error is `stickers-to-spell-word`: its gold runs a BFS over 2^15 states and hits `MemoryError` under the 2 GB cap. It fails the same way every run, so the problem is dropped and the cap is kept.

### Gold vs gold (`data/.slurm/gate_check.py`)

Each gold submitted 3× as the "submission" against a fresh pool, so gold timing is re-measured independently. 8,181 submissions per run.

| Workers | Verdicts | Re-checks | Sub/gold CPU ratio p50 / p99 / max | Throughput |
|---|---|---|---|---|
| 8 | 8,181 PASS | 0 | 1.00 / 1.32 / 2.32 | 74 sub/s |
| 16 | 8,181 PASS | 0 | 1.00 / 1.36 / 1.90 | 142 sub/s |

- Zero false `TOO_SLOW` at k = 3 for every floor tried (0, 0.02, 0.05, 0.1 s), even before the re-check. Floor kept at 0.05 s.
- Gold CPU time: p50 8 ms, p90 25 ms, p99 3.0 s, max 9.6 s. The tail motivated dropping golds above 3.3 s.
- Throughput is linear in workers (~9 sub/s per core). An RL step of 64 prompts × 8 samples = 512 responses takes ~7 s at W = 8, ~3.6 s at W = 16. `MALFORMED` responses never run, so real steps are faster.

### End-to-end smoke (Qwen3-4B-Base, no SFT, one B300)

**Base-model evaluation on `eval` (220 test problems, 1 sample).**

| Decoding | Max tokens | PASS | WRONG_ANSWER | RUNTIME_ERROR | TIMEOUT | MALFORMED |
|---|---|---|---|---|---|---|
| T = 1.0 | 2048 | 2 (0.9%) | 9 | 3 | 0 | 206 |
| greedy | 1024 | 27 (12.3%) | 59 | 9 | 4 | 121 |

- Greedy by difficulty: Easy 17/46 (37%), Medium 6/98 (6%), Hard 4/76 (5%). 73 of 220 responses were cut off while still thinking.
- `MALFORMED` (no fenced code block inside `<answer>`, under the original fenced-block rule) is the main failure at both temperatures. At T = 1.0, 95 of the 206 have code but no `<answer>`, and many responses drift into multilingual noise. The grader is left strict; SFT is meant to teach the format.
- No `TOO_SLOW` verdicts and no infra errors in either run.

**RL smoke** (`train_rl --total-steps 3 --train-batch-size 16 --n-samples 4 --max-response-length 1024`, `VERIFIER_WORKERS=4`, started from the base model):

- 3 steps completed and the final model was saved. Train reward 0.016.
- Reward time 0.2 s per step, against 18–33 s of generation, so the verifier is not the bottleneck.
- About half of the responses hit the 1,024-token limit.
- Greedy validation on `rl_val` (125 train-split problems): reward 0.325, malformed 0.49, `TOO_SLOW` 0, infra errors 0.

**rl_val 32.5% vs eval 12.3% (both greedy, 1,024 tokens).** Part of the gap is the difficulty mix: rl_val is 29% Easy / 19% Hard, while eval is 21% Easy / 35% Hard. If eval's per-difficulty rates are applied to rl_val's mix, the expected score is about 15%. The remaining ~17 points are consistent with the train problems being older and more likely seen in pretraining. Report results on `eval` only.

### Environment notes

- **B300 (sm_103a).** Triton's bundled `ptxas` does not support this architecture. Jobs on B300 set `TRITON_PTXAS_PATH=/share/goyal/ayf7/tools/cuda-13.0/bin/ptxas`, a copy of the CUDA 13.0 `ptxas`.
- **verl install.** The shared env's editable verl install points to the main checkout's `verl/`, not this worktree's. The reward function is still loaded from this worktree by path (`custom_reward_function.path`), so the LeetCode code under test is this branch's.
