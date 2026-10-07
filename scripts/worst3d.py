"""Worst measured 3D validation items (ratio = measured / truth) with solver details."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2
from pipeline.scoring import RESULTS, mra
df = m2.load_split("validation"); cache = RESULTS / "cache/validation"
parsed = m2.load_parsed(df, cache)
sol = pd.read_csv(cache / f"{sys.argv[1]}_solutions.csv").merge(df, on="id")
s = sol[sol.category.isin(["S3", "D3"]) & sol.value.notna()].copy()
s["ratio"] = (s.value / s.answer).round(2); s["mra"] = mra(s.value, s.answer)
s["q"] = [parsed[i]["target_quantity"] for i in s.id]
s["refq"] = [parsed[i]["ref_quantity"] for i in s.id]
print("mean MRA measured:", s.groupby("category").mra.mean().round(3).to_dict())
print("by target quantity:"); print(s.groupby(["category", "q"]).mra.agg(["mean", "count"]).round(2).to_string())
cols = ["video_id", "question", "answer", "value", "ratio", "q", "refq", "f_over_w", "f_source", "tgt_depth"]
print(s.sort_values("mra")[[c for c in cols if c in s]].head(25).to_string(index=False))
