"""One-at-a-time sweep of solver measurement choices on cached validation tracks (CPU only).

Reports S2/D2 MRA (measurement with VLM fallback) and MRA by target quantity, so each
choice can be judged on the items it actually affects.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.error_analysis import target_kind  # noqa: E402
from experiments.measure_2d import load_parsed, stage_solve  # noqa: E402
from pipeline import solver  # noqa: E402
from pipeline.data import load_split  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402

cache = RESULTS / "cache" / "validation"
val = load_split("validation")
df = val[val["category"].isin(["S2", "D2"])].copy()
df["target"] = df["question"].map(target_kind)
parsed = load_parsed(df, cache)
vlm = pd.read_csv(RESULTS / "validation_predictions" / "Qwen3-VL-4B_video_f16_px640.csv")
vlm = df[["id"]].merge(vlm[["id", "parsed_value"]], on="id", how="left")["parsed_value"].values
DEFAULT = dict(solver.SETTINGS)

VARIANTS = [("default", {})] + [
    (f"{k}={v}", {k: v}) for k, vals in {
        "width": ["minor"],
        "smooth_s": [0.5, 1.0],
        "mean_speed": ["path", "chord"],
        "straight_ratio": [0.6, 0.9],
        "accel_half_s": [0.5, 0.75],
        "min_ref_motion_px": [0],
    }.items() for v in vals
]

rows = []
for name, change in VARIANTS:
    solver.SETTINGS.clear()
    solver.SETTINGS.update({**DEFAULT, **change})
    sol = df[["id"]].merge(stage_solve(df, parsed, cache)[["id", "value"]], on="id", how="left")
    pred = np.where(sol["value"].notna(), sol["value"], vlm)
    per = mra(pred, df["answer"].values)
    r = {"variant": name, "solved": int(sol["value"].notna().sum())}
    for c in ["S2", "D2"]:
        r[c] = per[(df["category"] == c).values].mean() * 100
    r["avg2D"] = (r["S2"] + r["D2"]) / 2
    for t in ["size", "velocity", "acceleration", "distance"]:
        r[t] = per[(df["target"] == t).values].mean() * 100
    rows.append(r)

print(pd.DataFrame(rows).round(1).to_string(index=False))
print("\nitems per target:", df["target"].value_counts().to_dict())
