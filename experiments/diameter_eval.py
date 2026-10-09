"""Diameter measurement choice (affects diameter answers AND diameter priors' scale).
Validation: official-style MRA per category with the VLM fallback, and diameter items' direction.
Test (label-free): variant disagreement of measured items in clips that use diameters."""
import re
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2  # noqa: E402
import pipeline.solver as s2  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402


def base(vid: str) -> str:
    return re.sub(r"(_segmented|(?<=\d)[sx]|(?<=\d[a-z])[sx])$", "", vid)


MODES = ["eq", "minor", "bbox_min", "bbox_mean"]
out = {}
for split in ["validation", "test"]:
    df = m2.load_split(split); cache = RESULTS / "cache" / split
    parsed = m2.load_parsed(df, cache)
    uses = df.id.map(lambda i: "diameter" in (parsed.get(i, {}).get("target_quantity"), parsed.get(i, {}).get("ref_quantity")))
    sub = df[df.video_id.isin(df.loc[uses, "video_id"])]  # only clips touching diameters can change
    for m in MODES:
        s2.SETTINGS["diameter"] = m
        out[(split, m)] = m2.stage_solve(sub, parsed, cache).set_index("id")["value"]
    s2.SETTINGS["diameter"] = "eq"
    out[split] = (sub.set_index("id"), parsed)

d, parsed = out["validation"]
vlm = pd.read_csv(RESULTS / "validation_predictions/Qwen3-VL-4B_video_f16_px448.csv")
vlm = vlm.set_index(vlm.columns[0])["parsed_value"]
full_df = m2.load_split("validation").set_index("id")
base_val = pd.read_csv(RESULTS / "validation_predictions/val_measure_v19.csv")
base_val = base_val.set_index(base_val.columns[0])["parsed_value"]
for m in MODES:
    v = out[("validation", m)]
    pred = base_val.copy()
    pred.loc[v.index] = v.fillna(vlm.reindex(v.index))
    cats = {c: round(mra(pred[full_df.category == c], full_df.answer[full_df.category == c]).mean() * 100, 1)
            for c in ["S2", "D2", "S3", "D3"]}
    dq = [i for i in v.index if parsed.get(i, {}).get("target_quantity") == "diameter" and np.isfinite(v[i])]
    r = v[dq] / d.answer[dq]
    print(f"{m:9s} avg {np.mean(list(cats.values())):.1f} {cats}  diameter items n={len(dq)} MRA "
          f"{mra(v[dq], d.answer[dq]).mean():.2f} median ratio {r.median():.2f}")

t, _ = out["test"]
t = t.assign(b=t.video_id.map(base))
for m in MODES:
    v = out[("test", m)]
    x = t.assign(v=v).dropna(subset=["v"])
    sp = x.groupby(["b", "question"]).v.agg(lambda s: s.max() / s.min() if len(s) > 1 else np.nan).dropna()
    print(f"test {m:9s}: measured {v.notna().sum()}, variant groups {len(sp)}, inconsistent (>1.25x) {(sp > 1.25).mean():.2f}")
