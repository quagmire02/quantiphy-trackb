"""Where does MRA get lost? Per-item breakdown of validation predictions.

For each run: MRA by source / inference type / target quantity / category, the signed log10
error (bias: over- vs under-estimation), order-of-magnitude misses (likely unit/scale confusion),
and the worst questions.

  python experiments/error_analysis.py Qwen3-VL-4B_video_f16_px640 ref_gpt-5.1
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402

pd.set_option("display.width", 200, "display.max_colwidth", 90)


def target_kind(q: str) -> str:
    q = q.lower()
    if "acceleration" in q or "/s^2" in q or "/s²" in q:
        return "acceleration"
    if re.search(r"speed|velocity|/s\b", q):
        return "velocity"
    if re.search(r"distance|displacement|travel", q):
        return "distance"
    return "size"


def load_run(name: str, val: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_csv(RESULTS / "validation_predictions" / f"{name}.csv")
    p = p.rename(columns={p.columns[0]: "id"})[["id", "parsed_value"]]
    df = val.merge(p, on="id", how="left")
    df["mra"] = mra(df["parsed_value"], df["answer"])
    pred = df["parsed_value"].abs().where(df["parsed_value"].abs() > 0)
    df["log_err"] = np.log10(pred / df["answer"].abs())  # + = overestimate
    return df


def table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by)
    return pd.DataFrame({
        "n": g.size(),
        "MRA": (g["mra"].mean() * 100).round(1),
        "zero_mra_%": (g["mra"].apply(lambda s: (s == 0).mean()) * 100).round(0),
        "median_log10_err": g["log_err"].median().round(2),
        "off_>=3x_%": (g["log_err"].apply(lambda s: (s.abs() >= np.log10(3)).mean()) * 100).round(0),
        "invalid": g["log_err"].apply(lambda s: s.isna().sum()),
    }).sort_values("MRA")


def main():
    val = load_split("validation")
    val["target"] = val["question"].map(target_kind)
    val["prior_kind"] = val["inference_type"].str[0].map({"S": "size prior", "D": "motion prior"})
    runs = sys.argv[1:] or ["Qwen3-VL-4B_video_f16_px640", "ref_gpt-5.1"]
    frames = {r: load_run(r, val) for r in runs}

    for name, df in frames.items():
        print(f"\n{'=' * 100}\n{name}   overall item-mean MRA {df['mra'].mean() * 100:.1f}\n{'=' * 100}")
        for by in ["category", "video_source", "inference_type", "target", ["category", "target"]]:
            print(f"\n-- by {by}")
            print(table(df, by).to_string())
        print("\n-- error size distribution (|log10 ratio|):")
        e = df["log_err"].abs()
        bins = pd.cut(e, [-0.001, np.log10(1.1), np.log10(1.5), np.log10(3), 1, 10], labels=[
            "<10%", "10-50%", "1.5-3x", "3-10x", ">=10x"])
        print(bins.value_counts(dropna=False).reindex(["<10%", "10-50%", "1.5-3x", "3-10x", ">=10x", np.nan]).to_string())
        print("\n-- 12 worst items")
        worst = df.assign(ratio=(df["parsed_value"] / df["answer"]).round(2)).sort_values(["mra", "log_err"], key=lambda s: s.abs() if s.name == "log_err" else s)
        worst = worst.sort_values(by="log_err", key=lambda s: -s.abs().fillna(99)).head(12)
        print(worst[["video_id", "category", "question", "prior", "answer", "parsed_value", "ratio"]].to_string(index=False))

    if len(frames) >= 2:
        a, b = runs[:2]
        both = frames[a][["id", "video_id", "category", "target", "mra"]].merge(
            frames[b][["id", "mra"]], on="id", suffixes=(f"_{a[:10]}", f"_{b[:10]}"))
        cols = [c for c in both.columns if c.startswith("mra_")]
        print(f"\n-- per-video MRA, {a} vs {b}")
        pv = both.groupby(["video_id", "category"])[cols].mean().mul(100).round(0)
        pv["n"] = both.groupby(["video_id", "category"]).size()
        print(pv.sort_values(cols[0]).to_string())
        print(f"\nitems where both score 0: {((both[cols[0]] == 0) & (both[cols[1]] == 0)).sum()} / {len(both)}")


if __name__ == "__main__":
    main()
