"""Question families by wording: size, answer source, validation MRA, test variant disagreement.
A family = what the question actually asks to measure, beyond the plain object extent."""
import re
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402

FAMILIES = [  # first match wins
    ("component (horizontal/vertical motion)", r"\b(horizontal|vertical)\w*\s+(velocity|speed|acceleration|displacement|distance)|\b(velocity|speed|acceleration|displacement)\b.*\b(horizontal|vertical)"),
    ("gap between surfaces (above/below/from..to)", r"\b(above|below|beneath|under|off|from the (ground|floor|water|table|desk))\b.*\b(height|distance|clearance|depth)|\b(height|distance|clearance|depth)\b.*\b(above|below|off the|from the (ground|floor|water|table|desk|surface))"),
    ("between two objects", r"\bbetween\b.*\band\b|\bbetween the two\b|\bdistance from .* to\b"),
    ("part of an object", r"(?:'s|’s)\s+\w+|\b(wheel|tire|tyre|door|window|handle|leg|arm|head|wing|lid|rim|cap|screen|opening|seat|roof|base|top|neck|tail|shoulder|label|logo|text|letter)s?\s+of\b|\b(rim|opening|lid|handle|screen|label|logo|lettering|text)\b"),
    ("span / breadth (wingspan, shoulder, arm)", r"wingspan|wing span|shoulder|arm span|breadth"),
    ("path (orbit, circular, travelled)", r"orbit|circular|path|travel+ed|total distance"),
    ("region (floor, lane, road, tile, wall)", r"\b(floor|lanes?|road|tiles?|wall|ground|court|field|street|sidewalk|ceiling|roof frame)\b"),
    ("thickness / depth", r"thickness|\bdepth\b|\bthick\b"),
    ("plain size", r"\b(length|width|height|diameter|radius|size|long|tall|wide)\b"),
    ("plain speed / velocity", r"\b(speed|velocity)\b"),
    ("plain acceleration", r"acceleration"),
    ("displacement", r"displacement"),
]


def family(q: str) -> str:
    q = q.lower()
    for name, pat in FAMILIES:
        if re.search(pat, q):
            return name
    return "other"


def base(v):
    return re.sub(r"(_segmented|(?<=\d)[sx]|(?<=\d[a-z])[sx])$", "", v)


v = m2.load_split("validation")
vp = pd.read_csv(RESULTS / "validation_predictions/val_measure_v20.csv")
v = v.merge(vp.rename(columns={vp.columns[0]: "id"})[["id", "parsed_value", "source"]], on="id")
v["fam"] = v.question.map(family)
v["mra"] = mra(v.parsed_value, v.answer)
t = m2.load_split("test")
tp = pd.read_csv(RESULTS / "submissions/submission_v11_sources.csv")[["id", "parsed_value", "source"]]
t = t.merge(tp, on="id")
t["fam"] = t.question.map(family)
t["b"] = t.video_id.map(base)
sp = t.groupby(["b", "question"]).parsed_value.agg(lambda s: s.max() / s.min() if len(s) > 1 else np.nan)
t = t.merge(sp.rename("spread").reset_index(), on=["b", "question"])
rows = []
for f in [n for n, _ in FAMILIES] + ["other"]:
    tv, vv = t[t.fam == f], v[v.fam == f]
    tw = tv.dropna(subset=["spread"])
    rows.append({"family": f, "test_n": len(tv), "test_share_%": round(100 * len(tv) / len(t), 1),
                 "test_measured_%": round(100 * (tv.source == "measure").mean(), 0) if len(tv) else np.nan,
                 "test_inconsistent_%": round(100 * (tw.spread > 1.25).mean(), 0) if len(tw) else np.nan,
                 "val_n": len(vv), "val_MRA": round(vv.mra.mean(), 2) if len(vv) else np.nan,
                 "val_points_lost": round((1 - vv.mra).sum(), 1)})
r = pd.DataFrame(rows)
print(r.to_string(index=False))
print("\nexamples (test):")
for f in r.family[:8]:
    ex = t[t.fam == f].question.drop_duplicates().head(4).tolist()
    print(f"  {f}:"); [print("     ", q) for q in ex]
