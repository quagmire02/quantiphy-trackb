"""Which part of the acceleration-prior check fires on test, and what did v1 answer for those clips?

For every clip with an acceleration prior: R^2 of quadratic / linear fits of the reference track
over the check window, the v1 measured answers of that clip, and implied person heights.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.measure_2d import load_parsed, object_keys  # noqa: E402
from pipeline.data import load_split  # noqa: E402
from pipeline.guards import PERSON  # noqa: E402
from pipeline.scoring import RESULTS  # noqa: E402
from pipeline.solver import SolveError, Track, _resolve, accel_check_window  # noqa: E402
from pipeline.units import to_si  # noqa: E402

pd.set_option("display.width", 220, "display.max_rows", 200)
split = sys.argv[1] if len(sys.argv) > 1 else "test"
cache = RESULTS / "cache" / split
df = load_split(split)
parsed = load_parsed(df, cache)
v1 = pd.read_csv(cache / ("measure_solutions.csv" if split == "test" else "measure2d_solutions.csv")).set_index("id")["value"]

rows = []
for vid, items in df.groupby("video_id"):
    recs = [parsed.get(i) for i in items["id"]]
    rec = next((r for r in recs if r and r["ok"]), None)
    if rec is None or rec["ref_quantity"] != "acceleration":
        continue
    tpath = cache / "tracks" / f"{vid}.npz"
    feats = dict(np.load(tpath)) if tpath.exists() else {}
    ref = object_keys(rec)[0]
    r2q = r2l = rel_se = np.nan
    if ref in feats:
        n = len(feats[ref])
        fps = float(items["fps"].iloc[0])
        tw = _resolve(rec["ref_time"], (n - 1) / fps)
        a, b = tw if tw else (0.0, (n - 1) / fps)
        try:
            tr = Track.from_feats(feats[ref], fps)
            r2q, r2l = tr.fit_quality(*accel_check_window(tw, a, b))
            rel_se = tr.accel_rel_se(*accel_check_window(tw, a, b))
        except SolveError:
            pass
    heights = []
    for i, r in zip(items["id"], recs):
        if r and r["target_quantity"] == "height" and any(PERSON.search(t) for t in r["targets"]) and np.isfinite(v1.get(i, np.nan)):
            heights.append(round(to_si(v1[i], r["target_unit"]), 2))
    rows.append({"video": vid, "dim": items["video_type"].iloc[0][1], "prior": items["prior"].iloc[0][:40],
                 "n_items": len(items), "n_measured_v1": int(v1.reindex(items["id"]).notna().sum()),
                 "r2_quad": round(r2q, 3), "r2_lin": round(r2l, 3), "gain": round(r2q - r2l, 3),
                 "rel_se": round(rel_se, 3), "person_h_v1": heights})
t = pd.DataFrame(rows)
reject = ~(t["rel_se"] <= 0.25)
print(f"{len(t)} clips with acceleration priors; relative-SE check rejects {reject.sum()} clips "
      f"({t.loc[reject, 'n_items'].sum()} items), keeps {(~reject).sum()} ({t.loc[~reject, 'n_items'].sum()} items)")
t["bad_person_h"] = t["person_h_v1"].map(lambda hs: any(not 0.8 <= h <= 2.4 for h in hs))
print("clips with an implausible v1 person height:", int(t["bad_person_h"].sum()),
      "| of those rejected by the SE check:", int((t["bad_person_h"] & reject).sum()))
print("clips with only plausible v1 person heights:", int((t["person_h_v1"].map(len) > 0 & ~t["bad_person_h"]).sum()),
      "| of those rejected:", int(((t["person_h_v1"].map(len) > 0) & ~t["bad_person_h"] & reject).sum()))
print("\n-- rejected:")
print(t[reject].sort_values("rel_se").to_string(index=False))
print("\n-- kept:")
print(t[~reject].sort_values("rel_se").to_string(index=False))
