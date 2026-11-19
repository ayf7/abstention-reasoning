"""Restore exponents lost from 16 leetcode_o2 statements. Each edit must match exactly once."""
import json, os, sys
from pipeline.tasks.leetcode import runtime_tests as RT
D = "data/leetcode_o2/problems"
FIX = {333: [("-228 <= nums1[i], nums2[i], nums3[i], nums4[i] <= 228", "-2^28 <= nums1[i], nums2[i], nums3[i], nums4[i] <= 2^28")],
       399: [("value at most than 230.", "value at most than 2^30.")], 467: [("in the range [1, 210].", "in the range [1, 2^10].")],
       499: [("1 <= nums[i] <\xa0216", "1 <= nums[i] <\xa02^16")], 527: [("0 <= image[i][j], color < 216", "0 <= image[i][j], color < 2^16")],
       599: [("0 <= nums[i] < 216", "0 <= nums[i] < 2^16")], 745: [("0 <= nums[i] < 216", "0 <= nums[i] < 2^16")],
       1150: [("in the range [1, 210].", "in the range [1, 2^10].")], 1280: [("0 <= deliciousness[i] <= 220", "0 <= deliciousness[i] <= 2^20")],
       1325: [("1 <= responses.length <= 102", "1 <= responses.length <= 10^2")], 1337: [("0 <= nums[i] < 210", "0 <= nums[i] < 2^10")],
       1841: [("in the range [1, 214].", "in the range [1, 2^14].")], 1872: [("doesn't exceed\xa0253-1", "doesn't exceed\xa02^53-1")],
       2156: [("1 <= nums[i] <= 230", "1 <= nums[i] <= 2^30")], 2234: [("1 <= nums[i] <= 220 - 1", "1 <= nums[i] <= 2^20 - 1")],
       2303: [("0 <= nums[i] < 230", "0 <= nums[i] < 2^30")]}
seen = {}
for split in ("primitives", "sft_train", "sft_val", "rl_train", "rl_val", "eval"):
    f = f"{D}/{split}.json"; rows = json.load(open(f)); n = 0
    for p in rows:
        for old, new in FIX.get(p["index"], []):
            assert p["problem"].count(old) == 1, (p["index"], old, p["problem"].count(old))
            p["problem"] = p["problem"].replace(old, new); n += 1
            seen.setdefault(p["index"], set()).add(split)
    if n:
        with open(f + ".tmp", "w") as o: json.dump(rows, o)
        os.replace(f + ".tmp", f)
    print(split, "rows", len(rows), "edited", n, flush=True)
    if split == "primitives": P = {p["index"]: p for p in rows}
assert set(seen) == set(FIX)
T = json.load(open("data/leetcode_o2/runtime_tests.json"))
for k in FIX:
    t = T[str(k)]
    try: RT.validate(P[k], t)
    except ValueError as e: print("REACH", k, e)
    print(k, sorted(seen[k]), [(s["name"], s["sizes"][-1], RT.stated_limit(P[k], s)) for s in t["series"] if RT.stated_limit(P[k], s)][:4], "limitations:", t["limitations"][:80])
