"""Gated looming depth (solver3d loom) and adaptive speed window (solver velocity_ici), each on/off:
validation category scores (VLM fallback where unmeasured) and label-free test variant agreement.

  python experiments/cpu_items_eval.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2  # noqa: E402
import pipeline.solver as s2  # noqa: E402
import pipeline.solver3d as s3  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402

VARIANTS = {"before (raw looming, fixed window)": ("raw", False), "+ gated looming": ("gated", False),
            "+ adaptive speed window": ("raw", True), "+ both": ("gated", True)}
splits = {}
for split in ["validation", "test"]:
    df = m2.load_split(split)
    splits[split] = (df, m2.load_parsed(df, RESULTS / "cache" / split), RESULTS / "cache" / split)
vlm = pd.read_csv(RESULTS / "validation_predictions/Qwen3-VL-4B_video_f16_px448.csv")
vlm = vlm.set_index(vlm.columns[0])["parsed_value"]
base_test = None
for name, (loom, ici) in VARIANTS.items():
    s3.SETTINGS["loom"], s2.SETTINGS["velocity_ici"] = loom, ici
    df, parsed, cache = splits["validation"]
    d = df.set_index("id")
    v = m2.stage_solve(df, parsed, cache).set_index("id")["value"]
    full = v.fillna(vlm.reindex(v.index))
    sc = mra(full, d.answer)
    cats = {c: round(sc[(d.category == c).values].mean() * 100, 1) for c in ["S2", "D2", "S3", "D3"]}
    df, parsed, cache = splits["test"]
    d = df.set_index("id")
    tv = m2.stage_solve(df, parsed, cache).set_index("id")["value"]
    b = d.video_id.str.replace(r"(_segmented|(?<=\d)[sxX]|(?<=\d[a-z])[sx])$", "", regex=True)
    x = d.assign(v=tv, b=b).dropna(subset=["v"])
    sp = x.groupby(["b", "question"]).v.agg(lambda s: s.max() / s.min() if len(s) > 1 else np.nan).dropna()
    if base_test is None:
        base_test = tv
    ch = int((~np.isclose(tv.fillna(-1), base_test.fillna(-1))).sum())
    print(f"{name:36s} val avg {np.mean(list(cats.values())):.1f} {cats} | test measured {int(tv.notna().sum())}, "
          f"variant disagreement {(sp > 1.25).mean():.3f}, changed vs before {ch}", flush=True)
