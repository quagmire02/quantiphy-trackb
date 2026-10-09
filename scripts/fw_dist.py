"""f/w from reliable calibrations (size priors) vs MoGe-2 f/w, by video source and variant (test+val)."""
import re, sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2
from pipeline.scoring import RESULTS
rows = []
for split, name in [("test", "measure_v20"), ("validation", "measure_v20")]:
    df = m2.load_split(split); cache = RESULTS / "cache" / split
    parsed = m2.load_parsed(df, cache)
    sol = pd.read_csv(cache / f"{name}_solutions.csv").merge(df, on="id")
    sol = sol[sol.category.isin(["S3", "D3"])]
    for vid, g in sol.groupby("video_id"):
        refq = next((parsed[i]["ref_quantity"] for i in g.id if i in parsed), None)
        cal = g[(g.f_source == "prior")].f_over_w.median() if "f_source" in g else np.nan
        mf = m2.load_moge_f(cache, vid)
        gj = cache / "ground" / f"{vid}.json"
        w = json.loads(gj.read_text())["shape"][1] if gj.exists() else np.nan
        var = "seg" if re.search(r"(_segmented|\d[a-z]?s)$", vid) else ("x" if re.search(r"\d[a-z]?x$", vid) else "orig")
        rows.append((split, vid, vid.split("_")[0], var, refq, cal, (mf / w) if mf else np.nan))
r = pd.DataFrame(rows, columns=["split", "vid", "src", "var", "refq", "fw_cal", "fw_moge"])
size = r[r.refq.isin(["diameter", "length", "height", "width"])]
print("reliable calibration (size priors) f/w:")
print(size.groupby(["src", "var"]).fw_cal.describe()[["count", "25%", "50%", "75%"]].round(2).to_string())
print("\nMoGe f/w (all 3D clips):")
print(r.groupby(["src", "var"]).fw_moge.describe()[["count", "25%", "50%", "75%"]].round(2).to_string())
both = size.dropna(subset=["fw_cal", "fw_moge"])
both = both.assign(ratio=both.fw_moge / both.fw_cal)
print("\nMoGe / reliable calibration, by source x variant:")
print(both.groupby(["src", "var"]).ratio.describe()[["count", "25%", "50%", "75%"]].round(2).to_string())
