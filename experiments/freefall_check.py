"""Label-free physics check: under a gravity-like prior with a window starting at 0 ("before 0.6s"),
an object dropped from rest has speed g*t at time t. Compare measured answers to that for questions
asking the speed of the prior's own object inside its window. Not ground truth (the release instant may
differ from 0), but a consistent physical yardstick across solver versions.

  python experiments/freefall_check.py --split test --solutions measure_v8 measure_v9 measure_v10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.measure_2d import load_parsed  # noqa: E402
from pipeline.data import load_split  # noqa: E402
from pipeline.guards import _same_object  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402
from pipeline.units import from_si, to_si  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--solutions", nargs="+", required=True)
args = ap.parse_args()
cache = RESULTS / "cache" / args.split
df = load_split(args.split)
parsed = load_parsed(df, cache)
rows = []
for r in df.itertuples():
    rec = parsed.get(r.id)
    if not rec or not rec["ok"] or rec["ref_quantity"] != "acceleration" or rec["target_quantity"] != "speed":
        continue
    g = abs(to_si(rec["prior_value"], rec["prior_unit"]))
    rt, tt = rec["ref_time"], rec["target_time"]
    if not (9.0 <= g <= 10.5 and rt and rt[0] == 0 and tt and abs(tt[1] - tt[0]) < 1e-6
            and 0 < tt[0] <= rt[1] and len(rec["targets"]) == 1 and _same_object(rec["targets"][0], rec["ref_object"])):
        continue
    rows.append({"id": r.id, "video": r.video_id, "t": tt[0], "physics": from_si(g * tt[0], rec["target_unit"])})
d = pd.DataFrame(rows)
print(f"{len(d)} free-fall speed questions in {d['video'].nunique() if len(d) else 0} clips")
for name in args.solutions:
    v = pd.read_csv(cache / f"{name}_solutions.csv").set_index("id")["value"]
    m = d["id"].map(v)
    ok = m.notna()
    print(f"  {name:12s} measured {ok.sum():3d}/{len(d)}  MRA vs physics on measured: "
          f"{mra(m[ok], d.loc[ok, 'physics']).mean() * 100 if ok.any() else float('nan'):5.1f}")
