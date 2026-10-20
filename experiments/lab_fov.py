"""Lab camera field of view, from validation (allowed: validation answers for development).

(1) GT-implied: for every measured lab 3D size answer, L = px * Z / f, so the f that would make it
    exact is f_used * value / truth -> horizontal FOV per item; the median over lab clips.
(2) Sweep: validation S3/D3 scores (VLM fallback where unmeasured) with solver3d.SETTINGS["lab_fov_deg"]
    set to each candidate, plus the share of lab answers that change.
Test (label-free): frame sizes of lab clips (one rig -> one resolution) and the prior-calibrated FOV.

  python experiments/lab_fov.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2  # noqa: E402
import pipeline.solver3d as s3  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402

SIZE = {"length", "width", "height", "diameter", "size", "thickness"}
fov = lambda f, w: float(np.degrees(2 * np.arctan(w / 2 / f)))  # noqa: E731

df = m2.load_split("validation")
d3 = df[df.category.isin(["S3", "D3"])]
cache = RESULTS / "cache/validation"
parsed = m2.load_parsed(d3, cache)
s3.SETTINGS["lab_fov_deg"] = None
base = m2.stage_solve(d3, parsed, cache).merge(d3, on="id")
rows = []
for r in base[(base.video_source == "lab") & base.value.notna()].itertuples():
    if parsed[r.id]["target_quantity"] in SIZE and np.isfinite(r.f_px):
        w = json.loads((cache / "ground" / f"{r.video_id}.json").read_text())["shape"][1]
        rows.append((r.video_id, fov(r.f_px * r.value / r.answer, w), fov(r.f_px, w)))
g = pd.DataFrame(rows, columns=["vid", "fov_true", "fov_used"])
print(f"(1) GT-implied lab FOV over {len(g)} size answers in {g.vid.nunique()} clips: median {g.fov_true.median():.1f} deg "
      f"(per-clip medians {g.groupby('vid').fov_true.median().round(1).to_dict()}); the calibration used "
      f"{g.fov_used.median():.1f} deg")

vlm = pd.read_csv(RESULTS / "validation_predictions/Qwen3-VL-4B_video_f16_px448.csv")
vlm = vlm.set_index(vlm.columns[0])["parsed_value"]
d = d3.set_index("id")
lab = d.video_source == "lab"
print("(2) sweep (validation; 'lab' = lab 3D items only):")
for cand in [None, 70, 75, 78, 80, 82, 84, 86, 88, 90, 95]:
    s3.SETTINGS["lab_fov_deg"] = cand
    v = m2.stage_solve(d3, parsed, cache).set_index("id")["value"]
    full = v.fillna(vlm.reindex(v.index))
    sc = mra(full, d.answer)
    cats = {c: round(sc[(d.category == c).values].mean() * 100, 1) for c in ["S3", "D3"]}
    print(f"   fov {str(cand):>4}: S3 {cats['S3']}  D3 {cats['D3']}  lab items {sc[lab.values].mean() * 100:.1f} "
          f"(n={int(lab.sum())}, measured {int(v[lab].notna().sum())})", flush=True)
s3.SETTINGS["lab_fov_deg"] = None

t = m2.load_split("test")
tl = t[(t.video_source == "lab") & t.category.isin(["S3", "D3"])]
shapes = {}
for vid in tl.video_id.unique():
    p = RESULTS / "cache/test/ground" / f"{vid}.json"
    if p.exists():
        shapes.setdefault(tuple(json.loads(p.read_text())["shape"]), 0)
        shapes[tuple(json.loads(p.read_text())["shape"])] += 1
print(f"test lab 3D clips: {tl.video_id.nunique()}; tracking frame shapes (h, w): {shapes}")
