"""Write prediction CSVs and score validation runs with the *official* evaluator.

Every experiment writes results/<split>_predictions/<name>.csv. Scoring validation
re-runs QuantiPhy/evaluator.py over the whole folder, so all experiments land in one
comparison table (results/val_mra/all_model_results.csv).
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.data import ROOT, VAL_CSV

# QUANTIPHY_RESULTS: a separate results tree (caches, predictions, submissions), e.g. for a run with a
# bigger VLM that must not overwrite the 4B caches.
RESULTS = Path(os.environ["QUANTIPHY_RESULTS"]) if os.environ.get("QUANTIPHY_RESULTS") else ROOT / "results"
EVALUATOR = ROOT / "QuantiPhy" / "evaluator.py"
THETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]  # same as evaluator.py


def write_predictions(df: pd.DataFrame, values, name: str, split: str, extra: dict | None = None):
    """`df` from load_split; `values` aligned with df rows. `id` must stay the first column."""
    out = pd.DataFrame({
        "id": df["id"].values,
        "video_id": df["video_id"].values,
        "video_type": df["video_type"].values,
        "inference_type": df["inference_type"].values,
        "question": df["question"].values,
        "parsed_value": np.asarray(values, dtype=float),
    })
    for k, v in (extra or {}).items():
        out[k] = list(v)
    folder = RESULTS / f"{split}_predictions"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv"
    out.to_csv(path, index=False)
    return path


def mra(pred, gt) -> np.ndarray:
    """Per-item MRA, identical to evaluator.py (abs() of prediction, NaN -> 0)."""
    pred = np.abs(np.asarray(pred, dtype=float))
    gt = np.asarray(gt, dtype=float)
    rel = np.abs(pred - gt) / np.abs(gt)
    return np.mean([np.nan_to_num(rel < 1 - t, nan=0.0) for t in THETAS], axis=0)


def quick_report(df: pd.DataFrame, values, name: str) -> dict:
    """In-process score with per-category breakdown + bootstrap CI (validation only)."""
    per_item = mra(values, df["answer"].values)
    cats = ["S2", "D2", "S3", "D3"]
    by_cat = {c: per_item[(df["category"] == c).values].mean() for c in cats}
    avg = np.mean(list(by_cat.values()))

    # Bootstrap over videos (items within a video are correlated).
    rng = np.random.default_rng(0)
    vids = df["video_id"].unique()
    boots = []
    for _ in range(1000):
        pick = rng.choice(vids, size=len(vids))
        idx = np.concatenate([np.flatnonzero(df["video_id"].values == v) for v in pick])
        sub_cat = df["category"].values[idx]
        boots.append(np.mean([per_item[idx][sub_cat == c].mean() for c in cats if (sub_cat == c).any()]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"{name:32s} avg {avg * 100:5.1f}  [95% CI {lo * 100:4.1f}-{hi * 100:4.1f}]  "
          + "  ".join(f"{c} {by_cat[c] * 100:5.1f}" for c in cats))
    return {"name": name, "avg": avg, **by_cat, "ci_lo": lo, "ci_hi": hi}


def official_score():
    """Run the official evaluator over all validation prediction files."""
    cmd = [sys.executable, str(EVALUATOR), str(RESULTS / "validation_predictions"),
           str(RESULTS / "val_mra"), "--gt_file", str(VAL_CSV)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    table = pd.read_csv(RESULTS / "val_mra" / "all_model_results.csv")
    cols = ["model", "mra_average", "mra_S2", "mra_D2", "mra_S3", "mra_D3", "invalid_percentage"]
    table = table[cols].sort_values("mra_average", ascending=False)
    for c in cols[1:6]:
        table[c] = (table[c] * 100).round(1)
    table["invalid_percentage"] = table["invalid_percentage"].round(1)
    print("\nOfficial evaluator (validation):")
    print(table.to_string(index=False))
    return table
