"""Audit of the measurement inputs (label-free, test + validation):
1) frame rate: video container fps / frame count vs the dataset's fps column;
2) how many pixels the measured objects span (mask precision matters most for small ones);
3) unsolved items: reason, and whether the clip's scale (reference) itself was fine."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2  # noqa: E402
from pipeline.geometry import F  # noqa: E402
from pipeline.scoring import RESULTS  # noqa: E402

SOL = sys.argv[1] if len(sys.argv) > 1 else "measure_v22"
for split in ["validation", "test"]:
    df = m2.load_split(split)
    cache = RESULTS / "cache" / split
    print(f"\n######## {split}")
    rows = []
    for vid, g in df.groupby("video_id"):
        cap = cv2.VideoCapture(g.video_path.iloc[0])
        rows.append((vid, float(g.fps.iloc[0]), cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        cap.release()
    v = pd.DataFrame(rows, columns=["vid", "fps_csv", "fps_file", "frames", "w", "h"])
    v["ratio"] = v.fps_file / v.fps_csv
    off = v[(v.ratio - 1).abs() > 0.02]
    print(f"1) fps: {len(v)} videos; file fps differs from the dataset by >2% in {len(off)}")
    if len(off):
        print(off.round(3).head(25).to_string(index=False))
    print("   resolution (longest side):", v[["w", "h"]].max(axis=1).value_counts().sort_index().to_dict())
    print("   duration (s) quantiles:", (v.frames / v.fps_csv).quantile([0, .5, 1]).round(2).to_dict())

    parsed = m2.load_parsed(df, cache)
    sol = pd.read_csv(cache / f"{SOL}_solutions.csv").merge(df, on="id")
    sizes = []
    for r in sol[sol.value.notna()].itertuples():
        rec = parsed[r.id]
        if rec["target_quantity"] not in {"length", "width", "height", "diameter", "size", "thickness"}:
            continue
        feats = dict(np.load(cache / "tracks" / f"{r.video_id}.npz"))
        k = m2.object_keys(rec)[1][0]
        f = feats[k][feats[k][:, F["present"]] == 1]
        px = np.median(np.maximum(f[:, F["x2"]] - f[:, F["x1"]], f[:, F["y2"]] - f[:, F["y1"]]))
        sizes.append((r.category, px))
    s = pd.DataFrame(sizes, columns=["cat", "px"])
    bins = pd.cut(s.px, [0, 16, 32, 64, 128, 1e9], labels=["<16", "16-32", "32-64", "64-128", ">128"])
    print("2) measured size answers by target size in tracking pixels (longest side 1024; SAM mask grid ~4 px):")
    print(pd.crosstab(s.cat, bins).to_string())

    un = sol[sol.value.isna()].copy()
    un["why"] = un.reason.fillna("").str.replace(r"\[.*\]|-?\d+\.\d+|\d+", "", regex=True).str.slice(0, 55)
    print(f"3) unsolved items: {len(un)} of {len(sol)}")
    print(un.groupby(["why"]).size().sort_values(ascending=False).head(14).to_string())
