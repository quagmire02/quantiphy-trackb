"""Sweep one 3D solver setting: validation S3/D3 MRA (VLM fallback) and label-free test variant
consistency of measured 3D items.

  python experiments/sweep3d.py min_f_sensitivity 0 0.2 0.4 0.6 0.8
"""
import re
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2  # noqa: E402
import pipeline.solver3d as s3  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402

key, values = sys.argv[1], [float(v) for v in sys.argv[2:]]


def base(vid: str) -> str:
    return re.sub(r"(_segmented|(?<=\d)[sx]|(?<=\d[a-z])[sx])$", "", vid)


data = {}
for split in ["validation", "test"]:
    df = m2.load_split(split); df = df[df.category.isin(["S3", "D3"])]
    cache = RESULTS / "cache" / split
    data[split] = (df.set_index("id"), m2.load_parsed(df, cache), cache, df)
vlm = pd.read_csv(RESULTS / "validation_predictions/Qwen3-VL-4B_video_f16_px448.csv")
vlm = vlm.set_index(vlm.columns[0])["parsed_value"]
old = s3.SETTINGS[key]
for val in values:
    s3.SETTINGS[key] = val
    d, parsed, cache, df = data["validation"]
    sol = m2.stage_solve(df, parsed, cache).set_index("id")
    full = sol["value"].fillna(vlm.reindex(sol.index))
    cats = {c: round(mra(full[d.category == c], d.answer[d.category == c]).mean() * 100, 1) for c in ["S3", "D3"]}
    d, parsed, cache, df = data["test"]
    sol = m2.stage_solve(df, parsed, cache).set_index("id")
    x = d.assign(v=sol["value"], b=d.video_id.map(base)).dropna(subset=["v"])
    sp = x.groupby(["b", "question"]).v.agg(lambda s: s.max() / s.min() if len(s) > 1 else np.nan).dropna()
    src = sol.get("f_source", pd.Series(dtype=object)).value_counts().to_dict()
    print(f"{key}={val}: val {cats}  test measured {x.shape[0]} (f sources {src}), "
          f"variant inconsistency {(sp > 1.25).mean():.3f} over {len(sp)} groups", flush=True)
s3.SETTINGS[key] = old
