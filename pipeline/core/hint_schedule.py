"""Difficulty-scheduled hint generation.

The warm-start problem this solves: sampling hint requests from a fixed global
policy produced an SFT set where 89% of examples never asked for a hint, and the
difficulty signal that did exist was weak and non-monotone (Level 2 asked *less*
often than Level 1). A model trained on that learns "don't ask", leaving RL with
almost no hint-requesting behaviour to shape.

The fix is to decide how many hints each problem gets *before* generating, from
that problem's measured difficulty rather than its nominal label:

  Phase 1  probe    generate with NO hints, k samples per problem. The empirical
                    pass rate (correct / k) is the difficulty measure. A problem
                    the teacher already solves 5/5 times unaided does not need a
                    hint; one it never solves needs several.
  Phase 2  schedule bucket problems by pass rate and map each bucket to a hint
                    count. Lower pass rate -> more hints.
  Phase 3  generate run with the per-problem schedule, resampling problems that
                    come back incorrect until a target fraction is answered
                    correctly (see `generate_until_target` in
                    pipeline/commands/inference.py, which owns the loop).

Phase 1 output is a "difficulty profile" keyed by prompt index, so it is computed
once per (model, split) and reused across schedules.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# Pass-rate upper bound -> hints to schedule. Read as "solved at most X of the
# time unaided -> give it N hints". Ordered easiest-last so the first match wins.
DEFAULT_BUCKETS: list[tuple[float, int]] = [
    (0.00, 4),   # never solved unaided
    (0.25, 3),
    (0.50, 2),
    (0.75, 1),
    (1.00, 0),   # reliably solved unaided -> no hint
]


@dataclass
class DifficultyProfile:
    """Per-problem empirical difficulty from a no-hint probe."""
    model: str
    split: str
    num_samples: int
    pass_rates: dict[int, float] = field(default_factory=dict)
    # Reasoning tokens spent per problem. Secondary difficulty signal, needed
    # because pass rate saturates: a strong teacher solves most problems 8/8, so
    # a large block ties at 1.0 and rank order inside it would otherwise be
    # arbitrary. Longer reasoning on an equally-solved problem means harder.
    effort: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "split": self.split,
            "num_samples": self.num_samples,
            # JSON object keys are strings; keep the round-trip explicit.
            "pass_rates": {str(k): v for k, v in self.pass_rates.items()},
            "effort": {str(k): v for k, v in self.effort.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DifficultyProfile":
        return cls(
            model=d["model"],
            split=d["split"],
            num_samples=d["num_samples"],
            pass_rates={int(k): float(v) for k, v in d["pass_rates"].items()},
            effort={int(k): float(v) for k, v in d.get("effort", {}).items()},
        )


def parse_buckets(spec: str) -> list[tuple[float, int]]:
    """Parse "0.0:4,0.25:3,0.5:2,0.75:1,1.0:0" into bucket pairs.

    Each entry is `max_pass_rate:num_hints`. Sorted ascending so the first
    bucket whose bound the pass rate falls at or below decides the hint count.
    """
    buckets = []
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        bound, hints = pair.split(":")
        buckets.append((float(bound.strip()), int(hints.strip())))
    if not buckets:
        raise ValueError(f"--hint-buckets parsed to nothing: {spec!r}")
    buckets.sort(key=lambda b: b[0])
    for bound, hints in buckets:
        if not 0.0 <= bound <= 1.0:
            raise ValueError(f"bucket bound {bound} outside [0, 1] in {spec!r}")
        if hints < 0:
            raise ValueError(f"negative hint count {hints} in {spec!r}")
    return buckets


def hints_for_pass_rate(pass_rate: float, buckets: list[tuple[float, int]]) -> int:
    """Hint count for one problem's pass rate."""
    for bound, hints in buckets:
        if pass_rate <= bound:
            return hints
    return buckets[-1][1]


def build_profile(records: list[dict], model: str, split: str, num_samples: int) -> DifficultyProfile:
    """Turn no-hint probe records into a difficulty profile.

    Expects each record to carry `index` and either `pass_rate` (already
    aggregated over samples) or `correct` (a single sample).
    """
    rates: dict[int, float] = {}
    effort: dict[int, float] = {}
    for r in records:
        idx = r["index"]
        if "pass_rate" in r:
            rates[idx] = float(r["pass_rate"])
        else:
            rates[idx] = 1.0 if r.get("correct") else 0.0
        if r.get("token_count"):
            effort[idx] = float(r["token_count"])
    return DifficultyProfile(model=model, split=split, num_samples=num_samples,
                             pass_rates=rates, effort=effort)


def schedule_from_profile(
    prompts: list[dict],
    profile: DifficultyProfile,
    buckets: list[tuple[float, int]],
    default_hints: int = 0,
) -> dict[int, int]:
    """Map prompt index -> scheduled hint count.

    Prompts missing from the profile (never probed) fall back to
    `default_hints` rather than silently becoming 0, so a partial probe is
    visible in the summary instead of quietly skewing the mix.
    """
    schedule = {}
    for p in prompts:
        idx = p["index"]
        if idx in profile.pass_rates:
            schedule[idx] = hints_for_pass_rate(profile.pass_rates[idx], buckets)
        else:
            schedule[idx] = default_hints
    return schedule


def summarize_schedule(schedule: dict[int, int], profile: DifficultyProfile | None = None) -> str:
    """Human-readable breakdown of the scheduled hint mix."""
    if not schedule:
        return "  (empty schedule)"
    total = len(schedule)
    counts: dict[int, int] = {}
    for n in schedule.values():
        counts[n] = counts.get(n, 0) + 1
    lines = [f"  scheduled {total} problems:"]
    for n in sorted(counts):
        lines.append(f"    {n} hint(s): {counts[n]:5d} ({counts[n] / total:5.1%})")
    hinted = sum(c for n, c in counts.items() if n > 0)
    lines.append(f"  hint-requesting fraction: {hinted}/{total} = {hinted / total:.1%}")
    mean = sum(n * c for n, c in counts.items()) / total
    lines.append(f"  mean hints per problem:   {mean:.2f}")
    if profile is not None:
        missing = total - sum(1 for i in schedule if i in profile.pass_rates)
        if missing:
            lines.append(f"  WARNING: {missing} problems absent from profile, defaulted")
    return "\n".join(lines)


def schedule_by_rank(
    prompts: list[dict],
    profile: "DifficultyProfile",
    target_hint_fraction: float,
    max_hints: int = 4,
) -> dict[int, int]:
    """Schedule hint counts by difficulty *rank*, hitting the target fraction exactly.

    Why not pass-rate thresholds: a k-sample probe yields only k+1 distinct pass
    rates, and one value routinely carries a large share of the mass (with a weak
    teacher, "solved 0 of 8" can be 30%+ of the set). Any value threshold either
    includes that whole block or excludes it, so the achieved hint fraction jumps
    in coarse steps and two different targets can produce identical buckets.

    Ranking sidesteps that: sort hardest-first, give the hardest
    `target_hint_fraction` of problems hints, and split those into `max_hints`
    equal-sized bands (hardest band gets `max_hints`, easiest gets 1). Ties are
    broken by index so the assignment is deterministic across runs.
    """
    if not 0.0 <= target_hint_fraction <= 1.0:
        raise ValueError(f"target_hint_fraction must be in [0, 1], got {target_hint_fraction}")
    if max_hints < 1:
        raise ValueError(f"max_hints must be >= 1, got {max_hints}")

    indices = [p["index"] for p in prompts]
    # Sort hardest-first. Unprobed problems sort as easiest (pass rate 1.0) so
    # they land in the no-hint tail rather than silently absorbing the largest
    # hint budget. Ties on pass rate -- and with a strong teacher most of the set
    # ties at 1.0 -- are broken by reasoning effort, descending, so reaching into
    # the tie block still picks the harder problems rather than an arbitrary
    # slice. Index last, to keep the order deterministic.
    ranked = sorted(
        indices,
        key=lambda i: (profile.pass_rates.get(i, 1.0), -profile.effort.get(i, 0.0), i),
    )

    n_hinted = round(len(ranked) * target_hint_fraction)
    schedule = {i: 0 for i in ranked}
    if n_hinted == 0:
        return schedule

    for pos, idx in enumerate(ranked[:n_hinted]):
        # band 0 = hardest slice -> max_hints hints.
        band = min(max_hints - 1, pos * max_hints // n_hinted)
        schedule[idx] = max_hints - band
    return schedule
