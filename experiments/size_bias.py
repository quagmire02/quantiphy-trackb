"""Plain size / speed answers on validation: systematic bias by dimension, footage, and object pixel size."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2  # noqa: E402
from experiments.question_families import family  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402

v = m2.load_split("validation"); cache = RESULTS / "cache/validation"
parsed = m2.load_parsed(v, cache)
sol = pd.read_csv(cache / "measure_v20_solutions.csv")
v = v.merge(sol, on="id")
v = v[v.value.notna()].copy()
v["fam"] = v.question.map(family)
v["q"] = [parsed[i]["target_quantity"] for i in v.id]
v["lr"] = np.log(v.value / v.answer)
v["mra"] = mra(v.value, v.answer)
v["dim"] = v.video_type.str[1].map({"2": "2D", "3": "3D"})
v = v[v.fam.isin(["plain size", "plain speed / velocity"])]


def summ(g):
    b = [np.median(np.random.default_rng(i).choice(g.lr, len(g))) for i in range(500)]
    return pd.Series({"n": len(g), "median_ratio": np.exp(g.lr.median()), "ci_lo": np.exp(np.percentile(b, 5)),
                      "ci_hi": np.exp(np.percentile(b, 95)), "too_small_%": 100 * (g.lr < -np.log(1.15)).mean(),
                      "too_big_%": 100 * (g.lr > np.log(1.15)).mean(), "mra": g.mra.mean()})


print("by quantity x 2D/3D:"); print(v.groupby(["q", "dim"]).apply(summ).round(2).to_string())
print("\nby video source:"); print(v.groupby(v.video_id.str.split("_").str[0]).apply(summ).round(2).to_string())
d2 = v[v.dim == "2D"].dropna(subset=["target_px", "ref_px"])
d2 = d2[d2.q.isin(["length", "width", "height", "diameter"])]
d2["tgt_bin"] = pd.cut(d2.target_px, [0, 20, 50, 150, 1e9], labels=["<20px", "20-50", "50-150", ">150"])
d2["ref_bin"] = pd.cut(d2.ref_px, [0, 20, 50, 150, 1e9], labels=["<20px", "20-50", "50-150", ">150"])
print("\n2D sizes by target pixel size:"); print(d2.groupby("tgt_bin", observed=True).apply(summ).round(2).to_string())
print("\n2D sizes by reference pixel size:"); print(d2.groupby("ref_bin", observed=True).apply(summ).round(2).to_string())
