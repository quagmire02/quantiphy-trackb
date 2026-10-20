"""Careful old-vs-new data comparison: empty values and whitespace normalised; validation matched by
(video_id, question) so removed rows and changed answers show up."""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD, NEW = ROOT / "data", ROOT / "data_new"


def norm(s: pd.Series) -> pd.Series:
    return s.map(lambda v: "" if pd.isna(v) else re.sub(r"\s+", " ", str(v)).strip())


a = pd.read_parquet(OLD / "test/test_dataset.parquet")
b = pd.read_parquet(NEW / "test/test_dataset.parquet")
print("== test")
for col in a.columns:
    x, y = norm(a[col]), norm(b[col])
    d = x != y
    if d.any():
        print(f"  {col}: {int(d.sum())} rows changed")
        for i in d[d].index[:6]:
            print(f"     id {i + 1} {a.video_id[i]}: {x[i][:120]!r}\n        -> {y[i][:120]!r}")
        if col == "depth_info":
            nums = lambda s: re.findall(r"\d*\.?\d+", s)  # noqa: E731
            numeric = [i for i in d[d].index if nums(x[i]) != nums(y[i])]
            print(f"     depth_info rows whose NUMBERS changed: {len(numeric)} in videos "
                  f"{sorted(set(a.video_id[numeric]))[:20]}")

ta = pd.read_csv(OLD / "template/quantiphy_submission_template.csv")
tb = pd.read_csv(NEW / "template/quantiphy_submission_template.csv")
print("\n== template vs new parquet (must line up for submission):")
for col in ["video_id", "question", "video_type", "depth_info", "fps"]:
    src = "ground_truth_prior" if col == "prior" else col
    if col in tb and col in b:
        print(f"  {col}: template != new parquet in {int((norm(tb[col]) != norm(b[col])).sum())} rows "
              f"(old template != new template in {int((norm(ta[col]) != norm(tb[col])).sum())})")
print("  ids:", tb.id.min(), tb.id.max(), "parsed_value all empty:", tb.parsed_value.isna().all())

va = pd.read_csv(OLD / "validation/validation_dataset.csv")
vb = pd.read_csv(NEW / "validation/validation_dataset.csv")
print(f"\n== validation: old {len(va)} rows, new {len(vb)} rows; id column old {va.columns[0]!r} new {vb.columns[0]!r}")
key = ["video_id", "question"]
for df in (va, vb):
    df["video_id"] = df.video_id.str.strip()
    df["question"] = norm(df.question)
m = va.merge(vb, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
print("  only in old:")
print(m[m._merge == "left_only"][key + ["ground_truth_posterior_old"]].to_string(index=False))
print("  only in new:")
print(m[m._merge == "right_only"][key + ["ground_truth_posterior_new"]].to_string(index=False))
both = m[m._merge == "both"]
for col in ["ground_truth_posterior", "ground_truth_prior", "depth_info", "video_type", "fps"]:
    d = norm(both[f"{col}_old"]) != norm(both[f"{col}_new"])
    if d.any():
        print(f"  {col} changed in {int(d.sum())} rows:")
        print(both[d][key + [f"{col}_old", f"{col}_new"]].head(10).to_string(index=False))
ids_old, ids_new = va.iloc[:, 0], vb.iloc[:, 0]
print("  first-column ids identical for matched rows:",
      bool((both[f"{va.columns[0]}_old"] == both[f"{vb.columns[0]}_new"]).all()) if va.columns[0] == vb.columns[0] else "n/a")
