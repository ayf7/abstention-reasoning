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
RL reward  ──┐  │ problem bank: id → test inputs, expected outputs, gold code, gold time (lazy) │
             ├─►│ dispatcher ──► worker 0 (pinned) ──fork──► sandboxed child                    │
evaluate   ──┘  │            ──► worker 1 (pinned) ──fork──► sandboxed child                    │
  batch of      │            ──► … one worker per physical core                                │
  (id, code)    └───────────────────────────────────────────────────────────────────────────────┘
                   ◄── one verdict per submission
```

- **Problem bank.** Loaded from `primitives.json` at service start. Callers send only `(problem id, code)`, so RL parquet rows stay small.
- **Dispatcher.** The service process. Holds expected outputs, sends jobs to workers, compares outputs, applies the gate.
- **Workers.** Started with `spawn` (clean interpreters) before the bank is loaded, pinned with `sched_setaffinity`. Workers receive only code and test inputs. Expected outputs never enter a worker's memory, so the forked child cannot find them via `gc`, frame inspection, or raw memory reads.

## One submission

1. **Extract code**: last ```` ```python ```` block in `<answer>`. None → `MALFORMED`, nothing runs.
2. **Fork** (≈1 ms vs 12–23 ms for a fresh interpreter). The child immediately:
   - closes every inherited fd except its result pipe; stdin/stdout/stderr → `/dev/null`;
   - `setsid` (so the worker can kill the whole group);
   - rlimits: address space, CPU seconds, file size 0, open files, no core dumps; stack raised for deep recursion;
   - Landlock: read-only access to the Python installation, nothing else (blocks reading the dataset or repo);
   - seccomp: `EPERM` for exec, fork/clone (threads included), sockets, ptrace, kill, namespace and mount calls.
3. **Run**: exec the dataset preamble (`from typing import *`, `ListNode`, `TreeNode`, …) plus the submission. For each test, build a fresh `Solution()` and call the entry method with that test's arguments. Write each result as one canonical JSON line (linked lists and trees become tagged value lists). On an exception, write its type and stop.
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
- `t_gold` is measured lazily by the service on its own workers, the first time a problem appears (median of 3 runs), then cached. This always matches the hardware the service runs on; there is no calibration file to go stale.
- **Borderline re-check**: if a correct submission exceeds the limit, the same worker re-runs gold and the submission back to back and decides on that pair. This cancels clock-speed drift.
- With LeetCodeDataset's tiny inputs (median largest input 174 characters), the gate only catches runaway code. Catching slow algorithms needs stress-input generators, which come with dataset augmentation.

## Data

- Tests come from the dataset's `check(candidate)` function, parsed with `ast`: each `assert` gives the call arguments (source) and the expected value. `input_output` is not used because it is lossy (it drops `list_node(...)`/`tree_node(...)` wrappers).
- Expected values are evaluated once at data-build time in the trusted process and stored canonically.
- Every gold solution is run through the real sandbox at build time. Problems whose gold fails, or whose asserts use an unsupported form, are dropped and counted.
- Problems whose description says "in any order" are dropped for now: the dataset's tests compare exactly against one ordering, so correct answers in another order would be marked wrong.

## Deployment

**RL.** `verl/recipe/leetcode/reward_function.py` provides a batch scorer for verl's `batch` reward manager, which sends a whole step's responses in one call to the service. `train_rl` starts the service on cores carved from the job's own allocation and pins verl to the rest (Mode A). Mode B (service on a separate CPU job, reached via `VERIFIER_ADDR`) uses the same code.

**Evaluation.** `BaseTask.check_correctness_batch` (default: loop over `check_correctness`, so other tasks are unchanged). The LeetCode task overrides it to send the batch to the service. Results record verdict and CPU time per sample.

## Validation checklist

| Check | Pass criterion |
|---|---|
| Gold through harness | failures dropped and counted |
| Gold vs gold timing | ~0 false `TOO_SLOW` at chosen k / floor |
| Adversarial suite (fork bomb, infinite loop, memory bomb, file read/write, sockets, `sys.exit`, `os._exit`, monkeypatching, huge output, signal-catching loops) | contained, correct verdict, no leftover processes |
| Throughput | wall time per 1,024-submission batch at W = 8, 16 |

## Results

(Filled in as the hill climb proceeds.)
