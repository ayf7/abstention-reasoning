"""Plot eval legs on SCOPE.md's precision-accuracy plane.

Requests are scored as abstentions (SCOPE.md 1, correction 4), so a leg's
operating point is measured on the problems it answered without asking:

    coverage  phi = answered-with-zero-hints / all problems
    precision P   = correct-at-zero-hints / answered-with-zero-hints
    accuracy  A   = correct-at-zero-hints / all problems       (A = P * phi)

The always-answer diagonal (P = A) is where every full-coverage model lives.
Height above it is coverage, not skill -- the skill question is whether P
clears a reference model that answers everything, because random selection
holds precision at the base rate. Two references exist and mean different
things: --anchor is THIS model forced to answer (the AR denominator), while
--baseline is a separate always-answer model and measures what the abstention
machinery cost. Without either, the plot shows position but not lift, and
says so.

Arms are listed explicitly rather than globbed. p=0, p=0.15, p=0.3 and p=0.4 are
left out on purpose: p=0 requests forever, and the rest sit in the same
extinguished cluster as p=0.2, so they add points without adding information.

    R=artifacts/competition_math/hint_ablations/results
    python scripts/analysis/plot_precision_accuracy.py --baseline 0.566 \
        --results $R/eval_augmented_rl_p0.025_last.json \
                  $R/eval_augmented_rl_p0.05_last.json \
                  $R/eval_augmented_rl_p0.075_last.json \
                  $R/eval_augmented_rl_p0.1_last.json \
                  $R/eval_augmented_rl_p0.2_last.json \
        --extra artifacts/competition_math/abstention_verify/results/eval_rl_3b.json \
        --out precision_accuracy_v2_hint.svg
"""

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Small enough that four arms 0.009 apart in accuracy stay distinguishable.
MARKER_SIZE = 36


# Offsets in points, applied in order of increasing accuracy. The live arms are
# as little as 0.009 apart in accuracy -- about three points on this axis -- so
# their labels cannot all sit in the same place relative to their marker. Fanning
# them left / above / below / right is what keeps them apart without leader
# lines. The list cycles if a future sweep adds arms.
LIVE_LABEL_OFFSETS = [
    (-6, 1, "right", "center"),
    (0, 8, "center", "bottom"),
    (7, 7, "left", "bottom"),
    (9, -3, "left", "center"),
]
LABEL_SIZE = 7


def label_points(ax, pts, offsets):
    """Name each point beside its own marker."""
    for i, d in enumerate(sorted(pts, key=lambda d: d["accuracy"])):
        dx, dy, ha, va = offsets[i % len(offsets)]
        ax.annotate(penalty_of(d["label"]), (d["accuracy"], d["precision"]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=LABEL_SIZE, color="0.25", ha=ha, va=va)


def leg_point(path: Path) -> dict:
    """Coverage, precision and accuracy for one results file."""
    data = json.loads(path.read_text())
    details = data["details"]
    # A leg abstains in one of two ways depending on the method: the hint arms
    # ask for a hint, abstention_verify emits <abstain>. Both are abstentions
    # under SCOPE.md, so a row is an answer only if it did neither. Rows that are
    # neither committed nor abstained -- truncated or malformed -- stay in the
    # denominator as answers that were wrong, which is how the hint side already
    # counts a truncated no-hint rollout.
    zero = [r for r in details
            if (r.get("num_hints") or 0) == 0 and not r.get("abstained")]
    correct = sum(1 for r in zero if r["correct"])
    total = len(details)
    hints = [r.get("num_hints") or 0 for r in details]
    return {
        "label": path.stem,
        "total": total,
        "answered": len(zero),
        "correct": correct,
        "coverage": len(zero) / total,
        "precision": correct / len(zero) if zero else float("nan"),
        "accuracy": correct / total,
        "assisted_accuracy": data["metrics"]["accuracy"],
        "mean_hints": sum(hints) / total,
        "max_hints": max(hints),
    }


def penalty_of(label: str) -> str:
    m = re.search(r"_p([0-9.]+)_", label + "_")
    return m.group(1) if m else label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", required=True, help="Results JSON paths or globs")
    ap.add_argument("--out", default="precision_accuracy_v2_hint.svg")
    ap.add_argument("--anchor", type=float, default=None,
                    help="kappa-bar: the matched baseline's forced-answer accuracy")
    ap.add_argument("--anchor-label", default="$\\bar\\kappa$ = accuracy with requests banned")
    ap.add_argument("--title", default="competition_math / hint_ablations - hint_penalty sweep")
    ap.add_argument("--baseline", type=float, default=None,
                    help="Accuracy of an external always-answer model (e.g. plain RLVR). "
                         "Distinct from --anchor, which is THIS model forced to answer.")
    ap.add_argument("--baseline-label", default="simple baseline")
    ap.add_argument("--extra", default=None,
                    help="A results file from a different method, drawn as a star. "
                         "It is a separate model, so it is kept off the sweep's line.")
    ap.add_argument("--extra-label", default="abstention_verify")
    ap.add_argument("--saturated-coverage", type=float, default=0.95,
                    help="At or above this coverage the arm has stopped abstaining at all, "
                         "so it is an always-answer point rather than an operating point.")
    ap.add_argument("--degenerate-coverage", type=float, default=0.05,
                    help="Legs below this coverage are drawn as degenerate")
    args = ap.parse_args()

    paths = sorted({Path(p) for pattern in args.results for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"no results matched: {args.results}")
    legs = [leg_point(p) for p in paths]
    legs.sort(key=lambda d: d["coverage"])

    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    # Iso-coverage rays P = A / phi. Coverage is fixed along each ray, so a leg
    # that buys accuracy purely by answering more slides along one of these.
    for phi in (0.25, 0.5, 0.75):
        ax.plot([0, phi], [0, 1], color="0.88", lw=0.8, zorder=0)
        ax.annotate(f"$\\varphi$={phi}", (phi, 1.0), textcoords="offset points",
                    xytext=(2, -10), fontsize=7, color="0.6")

    # Always-answer diagonal: phi = 1, so P = A. Every RLVR model lives here.
    ax.plot([0, 1], [0, 1], color="0.55", lw=1.2, ls="--", zorder=1)
    ax.annotate("always answer ($\\varphi$=1)", (0.72, 0.72), rotation=45,
                fontsize=8, color="0.45", ha="center", va="bottom")

    if args.baseline is not None:
        # A different model, so it is not an anchor for AR -- it answers the
        # separate question of what the abstention machinery cost against simply
        # training the same budget without it. Drawn independently of --anchor:
        # either reference stands on its own.
        ax.scatter([args.baseline], [args.baseline], s=200, marker="*",
                   color="darkred", zorder=5, edgecolor="white", lw=0.6)
        ax.annotate(f"{args.baseline_label} = {args.baseline:.3f}",
                    (args.baseline, args.baseline), textcoords="offset points",
                    xytext=(11, -3), fontsize=8, color="darkred")

    if args.anchor is not None:
        # Always-answering has coverage 1, so its precision and accuracy are the
        # same number: it is the single point (kappa-bar, kappa-bar) on the
        # diagonal, not a line. Drawing it that way makes the trade legible --
        # an arm up and to the LEFT of it bought precision by giving up
        # accuracy, which is exactly what these arms did.
        ax.scatter([args.anchor], [args.anchor], s=110, marker="*", color="crimson",
                   zorder=5, edgecolor="white", lw=0.6)
        ax.annotate("always answer", (args.anchor, args.anchor),
                    textcoords="offset points", xytext=(8, -12), fontsize=8,
                    color="crimson")
        ax.axhline(args.anchor, color="crimson", lw=1.0, ls=":", zorder=1)
        ax.annotate(f"{args.anchor_label} = {args.anchor:.3f}", (0.02, args.anchor),
                    textcoords="offset points", xytext=(0, 4), fontsize=8, color="crimson")

    # Both ends of the price range stop being selective predictors: too cheap and
    # the model requests forever, too dear and it never requests at all. Neither
    # end is an operating point on the risk-coverage curve, so neither sets the
    # scale the curve is read at -- they are drawn, labelled and set aside.
    live = [d for d in legs
            if args.degenerate_coverage <= d["coverage"] < args.saturated_coverage]
    dead = [d for d in legs if d["coverage"] < args.degenerate_coverage]
    saturated = [d for d in legs if d["coverage"] >= args.saturated_coverage]

    colors = plt.cm.viridis([i / max(len(live) - 1, 1) for i in range(len(live))])

    # The line runs through the saturated arms too: they are the far end of the
    # same price sweep, and cutting it at the last selective arm hid where the
    # curve actually goes once requesting stops.
    chain = sorted(live + saturated, key=lambda d: d["coverage"])
    if len(chain) > 1:
        ax.plot([d["accuracy"] for d in chain], [d["precision"] for d in chain],
                color="0.7", lw=1.0, zorder=2)
    if live:
        for d, c in zip(live, colors):
            ax.scatter(d["accuracy"], d["precision"], s=MARKER_SIZE, color=c,
                       edgecolor="black", lw=0.5, zorder=3)
        label_points(ax, live, LIVE_LABEL_OFFSETS)
    for d in dead:
        ax.scatter(d["accuracy"], d["precision"], s=MARKER_SIZE, facecolor="none",
                   edgecolor="0.5", lw=0.9, zorder=3)
        ax.annotate(f"{penalty_of(d['label'])}\n({d['mean_hints']:.0f} requests/problem)",
                    (d["accuracy"], d["precision"]), textcoords="offset points",
                    xytext=(8, -10), fontsize=LABEL_SIZE, color="0.4")
    for d in saturated:
        ax.scatter(d["accuracy"], d["precision"], s=MARKER_SIZE, facecolor="none",
                   edgecolor="0.5", lw=0.9, zorder=3)
    if saturated:
        label_points(ax, saturated, [(0, -10, "center", "top")])

    if args.extra:
        # Not part of the price sweep -- a different method at n=1 -- so it is a
        # star like the baseline rather than a sweep circle, and is left out of
        # the chain. Labelled below-left, the one side of it the sweep's own
        # labels leave free.
        extra = leg_point(Path(args.extra))
        # It lands 0.007 in accuracy from the p=0.05 arm, so it goes behind the
        # sweep's markers and the chain: the arm on top of it stays readable.
        ax.scatter(extra["accuracy"], extra["precision"], s=200, marker="*",
                   color="tab:orange", edgecolor="white", lw=0.6, zorder=1.5)
        ax.annotate(args.extra_label, (extra["accuracy"], extra["precision"]),
                    textcoords="offset points", xytext=(-8, -9),
                    fontsize=LABEL_SIZE, color="tab:orange", ha="right", va="top")
        legs.append(extra)

    ax.set_xlabel("accuracy $A$ = correct at zero hints / all problems")
    ax.set_ylabel("precision $P$ = correct at zero hints / attempted at zero hints")
    ax.set_title(args.title, fontsize=11)
    ax.set_xlim(-0.015, 1)
    ax.set_ylim(-0.015, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.15, lw=0.6)
    if args.anchor is None and args.baseline is None:
        ax.annotate("no reference model: height above the diagonal\nis coverage, not skill",
                    (0.02, 0.03), fontsize=8, color="0.35")

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")

    hdr = f"{'leg':>10} {'N':>5} {'answered':>9} {'phi':>6} {'correct':>8} {'P':>6} {'A':>6} {'assisted':>9}"
    print(hdr)
    for d in legs:
        print(f"{penalty_of(d['label']):>10} {d['total']:>5} {d['answered']:>9} "
              f"{d['coverage']:>6.3f} {d['correct']:>8} {d['precision']:>6.3f} "
              f"{d['accuracy']:>6.3f} {d['assisted_accuracy']:>9.3f}")


if __name__ == "__main__":
    main()
