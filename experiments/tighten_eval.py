"""Effect of this round's fixes on one split (run after the findmissing/refine/geom stages):
validation: official-style category scores with the VLM fallback, toggling refined sizes and the
equal-depth fallback; test: coverage and label-free variant disagreement.

  python experiments/tighten_eval.py validation
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2  # noqa: E402
import pipeline.solver3d as s3  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402

split = sys.argv[1] if len(sys.argv) > 1 else "validation"
df = m2.load_split(split)
cache = RESULTS / "cache" / split
parsed = m2.load_parsed(df, cache)
d = df.set_index("id")
variants = {"none (tracks only)": (False, False), "+ equal depth": (False, True),
            "+ refined sizes": (True, False), "+ both": (True, True)}
vlm = None
if split == "validation":
    vlm = pd.read_csv(RESULTS / "validation_predictions/Qwen3-VL-4B_video_f16_px448.csv")
    vlm = vlm.set_index(vlm.columns[0])["parsed_value"]
for name, (refine, eq) in variants.items():
    m2.REFINE_SIZES, s3.SETTINGS["equal_depth_fallback"] = refine, eq
    v = m2.stage_solve(df, parsed, cache).set_index("id")["value"]
    if split == "validation":
        full = v.fillna(vlm.reindex(v.index))
        cats = {c: round(mra(full[d.category == c], d.answer[d.category == c]).mean() * 100, 1) for c in ["S2", "D2", "S3", "D3"]}
        print(f"{name:20s} avg {np.mean(list(cats.values())):.1f} {cats}  measured {v.notna().sum()}", flush=True)
    else:
        b = d.video_id.str.replace(r"(_segmented|(?<=\d)[sx]|(?<=\d[a-z])[sx])$", "", regex=True)
        x = d.assign(v=v, b=b).dropna(subset=["v"])
        sp = x.groupby(["b", "question"]).v.agg(lambda s: s.max() / s.min() if len(s) > 1 else np.nan).dropna()
        print(f"{name:20s} measured {v.notna().sum()}  variant disagreement {(sp > 1.25).mean():.3f} ({len(sp)} groups)", flush=True)
