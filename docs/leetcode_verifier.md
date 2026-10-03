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

1. **Extract code**: last ```` ```python ```` block in `<answer>`. None → `MALFORMED`, nothing runs.
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
| `MALFORMED` | no code block |
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

**RL.** `verl/recipe/leetcode/reward_function.py` provides a batch scorer for verl's `batch` reward manager, which sends a whole step's responses in one call to the service. `train_rl` starts the service on cores carved from the job's own allocation and pins verl to the rest (Mode A). The verifier takes the last `$VERIFIER_WORKERS` (default 8) physical cores; their hyperthread siblings are left idle. Mode B (service on a separate CPU job, `python -m pipeline.core.verifier.service --socket PATH`, reached via `VERIFIER_SOCKET`) uses the same code.

**Evaluation.** `BaseTask.check_correctness_batch` (default: loop over `check_correctness`, so other tasks are unchanged). The LeetCode task overrides it to send the batch to the service. Results record verdict and CPU time per sample.

## Validation checklist

| Check | Pass criterion |
|---|---|
| Gold through harness | failures dropped and counted |
| Gold vs gold timing | ~0 false `TOO_SLOW` at chosen k / floor |
| Adversarial suite (fork bomb, infinite loop, memory bomb, file read/write, sockets, `sys.exit`, `os._exit`, monkeypatching, huge output, signal-catching loops) | contained, correct verdict, no leftover processes |
| Throughput | wall time per 1,024-submission batch at W = 8, 16 |

## Results

### Data build (full LeetCodeDataset, 32-CPU job, 71 s wall)

| | Count |
|---|---|
| Kept | 2,727 (train 2,507, test 220) |
| Dropped: "in any order" | 115 |
| Dropped: gold exceeds 10 s CPU | 26 |
| Dropped: gold runtime error | 1 |

Splits: sft_train 676, sft_val 76, rl_train 1,629, rl_val 126, eval 220. (Before the `timeout_s / k` rule; see below.)

### Gold vs gold (`data/.slurm/gate_check.py`)

Each gold submitted 3× as the "submission" against a fresh pool, so gold timing is re-measured independently. 8,181 submissions per run.

| Workers | Verdicts | Re-checks | Sub/gold CPU ratio p50 / p99 / max | Throughput |
|---|---|---|---|---|
| 8 | 8,181 PASS | 0 | 1.00 / 1.32 / 2.32 | 74 sub/s |
| 16 | 8,181 PASS | 0 | 1.00 / 1.36 / 1.90 | 142 sub/s |

- Zero false `TOO_SLOW` at k = 3 for every floor tried (0, 0.02, 0.05, 0.1 s), even before the re-check. Floor kept at 0.05 s.
- Gold CPU time: p50 8 ms, p90 25 ms, p99 3.0 s, max 9.6 s. The tail motivated dropping golds above 3.3 s.
- Throughput is linear in workers (~9 sub/s per core). An RL step of 64 prompts × 8 samples = 512 responses takes ~7 s at W = 8, ~3.6 s at W = 16. `MALFORMED` responses never run, so real steps are faster.
