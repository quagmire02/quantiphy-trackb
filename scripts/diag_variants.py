"""Where 3D variant disagreement comes from: calibration (f/w differs across variants of a scene)
or the target measurement (f/w agrees, answers differ). Test, label-free."""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2
from pipeline.scoring import RESULTS

def base(v): return re.sub(r"(_segmented|(?<=\d)[sx]|(?<=\d[a-z])[sx])$", "", v)
df = m2.load_split("test"); cache = RESULTS / "cache/test"
parsed = m2.load_parsed(df, cache)
sol = pd.read_csv(cache / f"{sys.argv[1]}_solutions.csv").merge(df, on="id")
s = sol[sol.category.isin(["S3", "D3"]) & sol.value.notna()].copy()
s["b"] = s.video_id.map(base)
s["q"] = [parsed[i]["target_quantity"] for i in s.id]
s["refq"] = [parsed[i]["ref_quantity"] for i in s.id]
g = s.groupby(["b", "question"])
s["vspread"] = g.value.transform(lambda x: x.max() / x.min())
s["fspread"] = g.f_over_w.transform(lambda x: x.max() / x.min())
s["n"] = g.value.transform("size")
s = s[s.n > 1]
s["bad"] = s.vspread > 1.25
s["fbad"] = s.fspread > 1.25
print("items with a variant twin:", len(s), "| inconsistent:", s.bad.mean().round(2))
print(pd.crosstab([s.category, s.fbad.rename("f differs >25%")], s.bad.rename("answer differs >25%")).to_string())
print("\nby prior quantity (inconsistent share, f-differs share):")
print(s.groupby("refq")[["bad", "fbad"]].mean().round(2).to_string())
print("\nby f source mix:")
s["fsrc"] = g.f_source.transform(lambda x: "+".join(sorted(set(x.fillna("?")))))
print(s.groupby("fsrc")[["bad", "fbad"]].agg(["mean", "size"]).round(2).to_string())
print("\nvariant suffix pairs: f/w per variant for the worst calibration groups")
w = s[s.fbad].drop_duplicates("video_id")[["b", "video_id", "refq", "f_over_w", "f_source", "ref_depth"]]
print(w.sort_values(["b", "video_id"]).head(40).to_string(index=False))
