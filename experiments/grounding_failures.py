"""Which objects does grounding fail on (test, one category)? Phrase, frames tried, Qwen's replies,
and how many items each failure costs - to find general patterns (typos, instance numbering, tiny
objects, attribute-heavy phrases, ...).

  python experiments/grounding_failures.py --split test --category D2 --solutions measure_v12
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.scoring import RESULTS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--category", default="D2")
ap.add_argument("--solutions", default="measure_v12")
args = ap.parse_args()
cache = RESULTS / "cache" / args.split
df = load_split(args.split)
df = df[df["category"] == args.category]
sol = pd.read_csv(cache / f"{args.solutions}_solutions.csv")
d = df.merge(sol, on="id")
d = d[d["reason"].astype(str).str.startswith("not grounded")]
rows = []
for r in d.itertuples():
    missing = re.findall(r"'([^']+)'", r.reason)
    g = json.loads((cache / "ground" / f"{r.video_id}.json").read_text())["objects"]
    for key in missing:
        base, _, inst = key.partition("#")
        o = g.get(base, {})
        rows.append({"video": r.video_id, "key": key, "phrase": o.get("phrase", base),
                     "instance": inst or "0", "n_boxes": len(o.get("boxes", [])),
                     "conflict": (o.get("conflict") or {}).get("with", ""),
                     "replies": " | ".join(x["reply"][:45].replace("\n", " ") for x in o.get("tried", []))[:120],
                     "question": r.question[:60]})
f = pd.DataFrame(rows)
agg = f.groupby(["video", "key", "phrase", "instance", "n_boxes", "conflict", "replies"]).size().rename("items").reset_index()
pd.set_option("display.width", 250, "display.max_colwidth", 120, "display.max_rows", 200)
print(f"{len(d)} items, {agg['video'].nunique()} clips, {len(agg)} failing objects\n")
print(agg.sort_values("items", ascending=False).to_string(index=False))
