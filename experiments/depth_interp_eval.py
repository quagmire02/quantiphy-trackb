"""3D depth between timed distances: linear in time vs apparent-size scaling (validation, labels).
Reports MRA per category with the VLM fallback, plus measured-only MRA per target quantity."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2  # noqa: E402
import pipeline.solver3d as s3  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402

split = sys.argv[1] if len(sys.argv) > 1 else "validation"
df = m2.load_split(split); df = df[df.category.isin(["S3", "D3"])]
cache = RESULTS / "cache" / split
parsed = m2.load_parsed(df, cache)
res = {}
for mode in ["linear", "size"]:
    s3.SETTINGS["depth_interp"] = mode
    res[mode] = m2.stage_solve(df, parsed, cache).set_index("id")["value"]
s3.SETTINGS["depth_interp"] = "linear"
d = df.set_index("id")
if split == "validation":
    vlm = pd.read_csv(RESULTS / "validation_predictions/Qwen3-VL-4B_video_f16_px448.csv")
    vlm = vlm.set_index(vlm.columns[0])["parsed_value"]
    q = pd.Series({i: parsed[i]["target_quantity"] if i in parsed else "" for i in d.index})
    for mode, v in res.items():
        full = v.fillna(vlm.reindex(v.index))
        print(mode, {c: round(mra(full[d.category == c], d.answer[d.category == c]).mean() * 100, 1) for c in ["S3", "D3"]},
              "measured by quantity:", {k: round(mra(v[g.index], d.answer[g.index]).mean(), 2)
                                        for k, g in v[v.notna()].groupby(q[v.notna()])})
else:
    both = res["linear"].notna() & res["size"].notna()
    lr = np.abs(np.log(res["size"][both] / res["linear"][both]))
    print(f"test: {both.sum()} both solved; differ > 10%: {(lr > np.log(1.1)).mean():.2f}; solved linear "
          f"{res['linear'].notna().sum()} size {res['size'].notna().sum()}")
