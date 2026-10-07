"""Label-free: MoGe-2 focal length vs the prior-calibrated focal length, per 3D clip that solved."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split
from pipeline.scoring import RESULTS

for split, name in [("validation", "measure_v15"), ("test", "measure_v15")]:
    df = load_split(split); cache = RESULTS / "cache" / split
    sol = pd.read_csv(cache / f"{name}_solutions.csv").merge(df[["id", "video_id"]], on="id")
    sol = sol[sol.f_px.notna()]
    rows = []
    for vid, g in sol.groupby("video_id"):
        p = cache / "moge" / f"{vid}.json"
        if not p.exists():
            continue
        fm = [f for f in json.loads(p.read_text())["f_px"] if f and np.isfinite(f)]
        if not fm:
            continue
        rows.append((vid, float(g.f_px.median()), float(np.median(fm)), float(np.ptp(fm) / np.median(fm)),
                     float(g.f_over_w.median())))
    r = pd.DataFrame(rows, columns=["vid", "f_cal", "f_moge", "moge_spread", "fw_cal"])
    r["ratio"] = r.f_moge / r.f_cal
    lr = np.abs(np.log(r.ratio))
    print(f"\n{split}: {len(r)} clips  |log ratio| median {np.median(lr):.3f}  within 15%: {(lr < np.log(1.15)).mean():.2f}"
          f"  within 25%: {(lr < np.log(1.25)).mean():.2f}  within 50%: {(lr < np.log(1.5)).mean():.2f}")
    r["src"] = r.vid.str.split("_").str[0]
    print(r.groupby("src").apply(lambda g: pd.Series({"n": len(g), "med_ratio": g.ratio.median(),
          "w25": (np.abs(np.log(g.ratio)) < np.log(1.25)).mean()})).round(2).to_string())
    print("low calibrated f/w (<0.3):")
    print(r[r.fw_cal < 0.3].round(3).to_string(index=False))
