"""For clips whose speed/acceleration prior was rejected as 'barely moves': how far does the reference
move vs its own tracking noise?  python scripts/ref_motion_stats.py <split> <solutions> [category]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.measure_2d import load_parsed, object_keys  # noqa: E402
from pipeline.data import load_split  # noqa: E402
from pipeline.scoring import RESULTS  # noqa: E402
from pipeline.solver import Track  # noqa: E402

split, name = sys.argv[1], sys.argv[2]
cat = sys.argv[3] if len(sys.argv) > 3 else None
cache = RESULTS / "cache" / split
df = load_split(split)
if cat:
    df = df[df["category"] == cat]
parsed = load_parsed(df, cache)
sol = pd.read_csv(cache / f"{name}_solutions.csv")
d = df.merge(sol, on="id")
d = d[d["reason"].astype(str).str.contains("barely moves")].drop_duplicates("video_id")
for r in d.itertuples():
    rec = parsed[r.id]
    ref = object_keys(rec)[0]
    f = dict(np.load(cache / "tracks" / f"{r.video_id}.npz"))[ref]
    tr = Track.from_feats(f, float(r.fps))
    d2 = np.diff(tr.pts, 2, axis=0)
    sigma = float(np.max(1.4826 * np.median(np.abs(d2 - np.median(d2, 0)), 0) / np.sqrt(6))) if len(d2) >= 3 else np.nan
    print(f"{r.video_id:28s} prior '{r.prior[:40]}'  extent {tr.extent():6.1f}px  noise {sigma:5.2f}px  "
          f"extent/noise {tr.extent() / max(sigma, 1e-3):7.1f}  points {len(tr.t)}")
