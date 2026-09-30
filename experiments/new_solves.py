"""Items measured in solutions B but not in A (or whose value changed), for plausibility review.

  python experiments/new_solves.py --split test --a measure_v8 --b measure_v9 [--category D2]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--category", default=None)
args = ap.parse_args()
cache = RESULTS / "cache" / args.split
df = load_split(args.split)
if args.category:
    df = df[df["category"] == args.category]
a = pd.read_csv(cache / f"{args.a}_solutions.csv")[["id", "value", "reason"]]
b = pd.read_csv(cache / f"{args.b}_solutions.csv")[["id", "value", "reason"]]
d = df.merge(a, on="id").merge(b, on="id", suffixes=("_a", "_b"))
changed = d[~np.isclose(d["value_a"].fillna(-1), d["value_b"].fillna(-1), rtol=1e-6)]
print(f"{len(changed)} items changed ({changed['value_a'].isna().sum()} newly measured, "
      f"{changed['value_b'].isna().sum()} newly unmeasured) in {changed['video_id'].nunique()} clips")
pd.set_option("display.width", 230, "display.max_colwidth", 60, "display.max_rows", 400)
cols = ["video_id", "prior", "question", "value_a", "value_b"]
if "answer" in d:
    changed = changed.assign(truth=changed["answer"], mra_a=mra(changed["value_a"], changed["answer"]),
                             mra_b=mra(changed["value_b"], changed["answer"]))
    cols += ["truth", "mra_a", "mra_b"]
print(changed[cols].round(3).to_string(index=False))
