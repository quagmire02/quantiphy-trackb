"""From moge_eval.csv: is MoGe's *relative* depth good enough to transfer a known distance to another
object in the same frame?  Predict each given distance from another anchored object:
d_j = d_i * moge_j / moge_i, and compare with the given d_j (inputs only, no answers).
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.scoring import RESULTS  # noqa: E402

for split in sys.argv[1:] or ["validation", "test"]:
    p = RESULTS / "cache" / split / "moge_eval.csv"
    if not p.exists():
        continue
    r = pd.read_csv(p)
    d = r[(r["kind"] == "distance") & np.isfinite(r["moge"]) & (r["moge"] > 0)]
    preds = []
    for (vid, fi), g in d.groupby(["video", "frame"]):
        g = g.drop_duplicates("object")
        for (_, a), (_, b) in itertools.permutations(g.iterrows(), 2):
            preds.append({"video": vid, "ratio": (a["given"] * b["moge"] / a["moge"]) / b["given"],
                          "true_depth_ratio": b["given"] / a["given"]})
    q = pd.DataFrame(preds)
    absr = d["moge"] / d["given"]
    lr_abs = np.abs(np.log(absr))
    print(f"\n{split}: {len(d)} anchored objects, {len(q)} same-frame transfers from {q['video'].nunique() if len(q) else 0} clips")
    print(f"  absolute MoGe distance : within 10% {(lr_abs < np.log(1.1)).mean():.0%}, 25% {(lr_abs < np.log(1.25)).mean():.0%}, "
          f"2x {(lr_abs < np.log(2)).mean():.0%}")
    if len(q):
        lr = np.abs(np.log(q["ratio"]))
        print(f"  relative transfer      : within 10% {(lr < np.log(1.1)).mean():.0%}, 25% {(lr < np.log(1.25)).mean():.0%}, "
              f"2x {(lr < np.log(2)).mean():.0%}  (median error {np.expm1(np.median(lr)):.0%})")
        naive = np.abs(np.log(q["true_depth_ratio"]))  # baseline: assume the same distance as the anchor
        print(f"  baseline 'same distance as the known object': within 10% {(naive < np.log(1.1)).mean():.0%}, "
              f"25% {(naive < np.log(1.25)).mean():.0%}, 2x {(naive < np.log(2)).mean():.0%}")
