"""Failure inventory for one scoring category (label-free on test).

For every item of the category: measured or not, and why not (solver reason). Groups the reasons
into general failure classes, counts clips and items per class, and lists example questions, so
fixes can target whole classes rather than single questions.

  python experiments/category_inventory.py --split test --category D2 --solutions measure_v8
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.measure_2d import load_parsed  # noqa: E402
from pipeline.data import load_split  # noqa: E402
from pipeline.scoring import RESULTS  # noqa: E402

CLASSES = [  # (class, regex on the solver reason)
    ("object not found by grounding", r"not grounded"),
    ("prior acceleration unreliable", r"acceleration too uncertain|free fall"),
    ("prior object barely moves", r"barely moves"),
    ("clip scale rejected (person/vehicle guard)", r"person height|vehicle size in clip"),
    ("single item rejected by guard", r"vehicle width|x the prior"),
    ("region, not an object", r"region"),
    ("parse failed / unsupported prior", r"parse failed"),
    ("tracking gap at asked time", r"not tracked|tracked frames|not enough"),
    ("no track file", r"not tracked$"),
]


def classify(reason: str) -> str:
    if not isinstance(reason, str) or not reason:
        return "measured"
    for name, pat in CLASSES:
        if re.search(pat, reason):
            return name
    return "other: " + re.sub(r"[\d.\[\]'-]+", "", reason)[:50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--category", default="D2")
    ap.add_argument("--solutions", default="measure_v8")
    args = ap.parse_args()
    cache = RESULTS / "cache" / args.split
    df = load_split(args.split)
    df = df[df["category"] == args.category]
    parsed = load_parsed(df, cache)
    sol = pd.read_csv(cache / f"{args.solutions}_solutions.csv")
    d = df.merge(sol[["id", "value", "reason"]], on="id", how="left")
    d["cls"] = d["reason"].map(classify)
    d["prior_kind"] = d["id"].map(lambda i: (parsed.get(i) or {}).get("ref_quantity"))
    d["target_q"] = d["id"].map(lambda i: (parsed.get(i) or {}).get("target_quantity"))
    d["source"] = d["video_id"].str.extract(r"^(\w+?)_")[0]

    n = len(d)
    t = d.groupby("cls").agg(items=("id", "size"), clips=("video_id", "nunique")).sort_values("items", ascending=False)
    t["share_%"] = (100 * t["items"] / n).round(1)
    print(f"{args.split} {args.category}: {n} items, {d['video_id'].nunique()} clips\n")
    print(t.to_string())
    print("\n-- by data source (share measured):")
    print(d.assign(m=d["cls"] == "measured").groupby("source")["m"].agg(["size", "mean"]).round(2).to_string())
    print("\n-- by prior kind (share measured):")
    print(d.assign(m=d["cls"] == "measured").groupby("prior_kind")["m"].agg(["size", "mean"]).round(2).to_string())
    pd.set_option("display.width", 220, "display.max_colwidth", 70)
    for cls in t.index:
        if cls == "measured":
            continue
        ex = d[d["cls"] == cls].drop_duplicates("video_id").head(6)
        print(f"\n== {cls}: e.g.")
        print(ex[["video_id", "prior", "question", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
