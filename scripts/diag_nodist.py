"""Why 3D objects without a stated distance got no transferred one (label-free)."""
import sys, collections
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2
import pipeline.solver3d as s3
from pipeline.scoring import RESULTS

df = m2.load_split("test"); cache = RESULTS / "cache/test"
parsed = m2.load_parsed(df, cache)
sol = pd.read_csv(cache / "measure_v16_solutions.csv").merge(df, on="id")
bad = sol[sol.reason.fillna("").str.contains("no distance given")]
why = collections.Counter(); ex = collections.defaultdict(list)
for vid, it in bad.groupby("video_id"):
    feats = dict(np.load(cache / "tracks" / f"{vid}.npz")); fps = float(it.fps.iloc[0])
    entries = s3.parse_depth(it.depth_info.iloc[0]); moge = m2.load_moge(cache, vid) or {}
    anchored = {}
    for k, f in feats.items():
        if s3.match_depth(entries, k):
            try: anchored[k] = s3.Object3D(f, s3.match_depth(entries, k), fps)
            except s3.SolveError: pass
    keys = set()
    for r in it.itertuples():
        ref, tg = m2.object_keys(parsed[r.id]); keys |= {k for k in [ref] + tg if not s3.match_depth(entries, k)}
    for k in keys:
        if not moge: c = "no moge file"
        elif not any(k in d for d in moge.values()): c = "object absent in all moge frames"
        elif not anchored: c = "no object with a stated distance is tracked"
        else:
            spreads, n_est = [], 0
            for fi, d in moge.items():
                if k not in d: continue
                est = [o.distance_at(fi) * d[k] / d[a] for a, o in anchored.items() if a != k and a in d]
                est = [e for e in est if np.isfinite(e) and e > 0]
                n_est += len(est)
                if est: spreads.append(max(est) / min(est))
            c = "no known object in the same frames" if not n_est else \
                ("known objects disagree > 2x" if min(spreads) > 2 else "frames disagree > 2x")
        why[c] += 1; ex[c].append(f"{vid}:{k} (stated: {sorted({e['name'] for e in entries})[:4]})")
for c, n in why.most_common():
    print(f"{n:3d} objects  {c}"); [print("      ", e) for e in ex[c][:5]]
