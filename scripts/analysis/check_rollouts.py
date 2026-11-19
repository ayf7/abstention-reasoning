"""Summarise the newest rollout dump per arm: malformed rate is the health signal.

Arms are discovered from the artifacts tree rather than hardcoded, so an arm
added to a sweep already in flight shows up without editing this.
"""
import json, glob, os, re

RL = "/share/goyal/ayf7/artifacts/competition_math/hint/models/rl"
arms = sorted(
    (m.group(1), d) for d in glob.glob(RL + "/qwen2.5-3b-math-hint-p*")
    if (m := re.search(r"-p([0-9.]+)$", d))
)
out = []
for pen, d in arms:
    fs = sorted(glob.glob(d + "/rollouts/*.jsonl"),
                key=lambda x: int(os.path.basename(x).split(".")[0]))
    if not fs:
        out.append(f"p{pen}=nodata"); continue
    try:
        rows = [json.loads(l) for l in open(fs[-1])]
    except Exception:
        out.append(f"p{pen}=readerr"); continue
    if not rows:
        out.append(f"p{pen}=empty"); continue
    n = len(rows)
    mal = sum(1 for r in rows if r.get("malformed"))
    hint = sum(1 for r in rows if (r.get("num_hints") or 0) > 0)
    cor = sum(1 for r in rows if r.get("correct"))
    step = os.path.basename(fs[-1]).split(".")[0]
    out.append(f"p{pen}[s{step}]: mal {mal/n:.1%} hint {hint/n:.1%} corr {cor/n:.1%}")
print(" | ".join(out) if out else "no arms found")
