"""Diagnose 3D clips where no focal length reproduces the prior: predicted reference value vs f."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2
import pipeline.solver3d as s3
from pipeline.scoring import RESULTS
from pipeline.units import to_si

split = sys.argv[1] if len(sys.argv) > 1 else "test"
sol_name = sys.argv[2] if len(sys.argv) > 2 else "measure_v15"
df = m2.load_split(split); cache = RESULTS / "cache" / split
parsed = m2.load_parsed(df, cache)
sol = pd.read_csv(cache / f"{sol_name}_solutions.csv")
bad = sol[sol.reason.fillna("").str.contains("no focal length")].merge(df, on="id")
for vid, it in bad.groupby("video_id"):
    row = it.iloc[0]; rec = parsed[row.id]
    feats = dict(np.load(cache / "tracks" / f"{vid}.npz"))
    h, w = json.loads((cache / "ground" / f"{vid}.json").read_text())["shape"]
    n = len(next(iter(feats.values()))); fps = float(row.fps); dur = (n - 1) / fps
    ref_key, tgts = m2.object_keys(rec)
    entries = s3.parse_depth(row.depth_info)
    anchors = s3.match_depth(entries, ref_key)
    prior_si = abs(to_si(rec["prior_value"], rec["prior_unit"]))
    print(f"\n=== {vid} ({len(it)} items) prior: {row.prior!r}  ref={ref_key!r} q={rec['ref_quantity']} t={rec['ref_time']}")
    print("   depth:", " | ".join(str(row.depth_info).splitlines()[:6]))
    print("   anchors:", [(a['name'], a['t'], round(a['d'], 3)) for a in anchors] or "NONE (moge?)")
    try:
        moge = m2.load_moge(cache, vid)
        anch = anchors or (s3.transfer_anchors(ref_key, {}, moge or {}, fps))
        ob = s3.Object3D(feats[ref_key], anch, fps, motion=rec["ref_quantity"] not in s3.SIZE_QUANTITIES)
        tw = s3._resolve(rec["ref_time"], dur)
        pres = ob.present
        print(f"   ref tracked {len(pres)}/{n} frames, pixel extent {np.ptp(ob.u):.0f}x{np.ptp(ob.v):.0f}, dist range {ob.distances().min():.2f}-{ob.distances().max():.2f}")
        for fm in [0.1, 0.3, 1, 3, 10, 30]:
            try:
                v = s3.quantity_3d(rec["ref_quantity"], tw, [ob], s3.Camera(w, h, fm * w), dur)
                print(f"   f={fm:5.1f}w -> {v:.4g} (prior {prior_si:.4g})")
            except Exception as e:
                print(f"   f={fm:5.1f}w -> ERR {e}")
    except Exception as e:
        print("   ERR", type(e).__name__, e)
    print("   questions:", it.question.head(3).tolist())
