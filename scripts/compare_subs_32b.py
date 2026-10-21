"""32B submission vs v16: file integrity (why only 3,010 rows matched), then answer differences by
category, source and question, and a labelled check on the test questions that also appear in
validation (validation answers; development use)."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.scoring import mra  # noqa: E402

a = pd.read_csv("/mnt/d/quantiphy/results/submissions/submission_v16.csv")
b = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/submission_qwen3_32b.csv")
src = pd.read_csv("/mnt/d/quantiphy/results/submissions/submission_v16_sources.csv").set_index("id")
print("== integrity")
print("rows", len(a), len(b), "| ids equal:", (a.id.values == b.id.values).all())
for col in a.columns:
    if col == "parsed_value":
        continue
    d = a[col].astype(str) != b[col].astype(str)
    if d.any():
        print(f"  column {col}: {int(d.sum())} rows differ from v16's, e.g. id {b.id[d].iloc[0]}: "
              f"{str(a[col][d].iloc[0])[:70]!r} vs {str(b[col][d].iloc[0])[:70]!r}")
v = pd.to_numeric(b.parsed_value, errors="coerce")
print(f"  32B parsed_value: empty/non-numeric {int(v.isna().sum())}, zero {int((v == 0).sum())}, "
      f"negative {int((v < 0).sum())}, inf {int(np.isinf(v).sum())}")

test = load_split("test")
cat = test.category.values
va, vb = a.parsed_value.values, v.values
ratio = vb / va
print("\n== answers vs v16 by category (ratio = 32B / v16)")
for c in ["S2", "D2", "S3", "D3"]:
    m = cat == c
    r = ratio[m]
    print(f"  {c}: {m.sum()} items, identical {int(np.isclose(r, 1, rtol=1e-6).sum())}, "
          f"differ >10% {int((np.abs(np.log(r)) > np.log(1.1)).sum())}, >2x {int((np.abs(np.log(r)) > np.log(2)).sum())}")
s = src.loc[a.id - 1, "source"].values
print("\n== how the differences split by v16's answer source")
print(pd.crosstab(cat, s, values=(np.abs(np.log(ratio)) > np.log(1.1)), aggfunc="mean").round(2).to_string())

# validation answers for the test questions that also appear in validation (matched on video + question)
val = load_split("validation")
norm = lambda q: re.sub(r"\s+", " ", str(q)).strip().lower()  # noqa: E731
key = dict(zip(zip(val.video_id, val.question.map(norm)), val.answer))
ans = np.array([key.get((vi, norm(q)), np.nan) for vi, q in zip(test.video_id, test.question)])
has = np.isfinite(ans)
print(f"\n== labelled overlap: {has.sum()} test questions also in validation")
for c in ["S2", "D2", "S3", "D3"]:
    m = has & (cat == c)
    if m.any():
        print(f"  {c} (n={m.sum()}): v16 {mra(va[m], ans[m]).mean() * 100:.1f}   32B {mra(vb[m], ans[m]).mean() * 100:.1f}")
m = has & (np.abs(np.log(ratio)) > np.log(1.1))
t = pd.DataFrame({"id": a.id[m], "cat": cat[m], "video": test.video_id[m], "question": test.question[m].str[:60],
                  "truth": ans[m], "v16": np.round(va[m], 3), "32B": np.round(vb[m], 3), "v16_src": s[m],
                  "mra_v16": mra(va[m], ans[m]), "mra_32B": mra(vb[m], ans[m])})
t["gain_32B"] = t.mra_32B - t.mra_v16
print("\noverlap items that changed, worst for 32B first:")
print(t.sort_values("gain_32B").to_string(index=False))
