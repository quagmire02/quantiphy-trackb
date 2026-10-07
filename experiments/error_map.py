"""Error map of the frozen pipeline (one solutions file + the VLM fallback, as submitted).

Validation: log-error per item (labels) by category x quantity x answer source x video source.
Test (label-free): the same question on variants of one scene (captured_0013 / 0013s / 0013x,
simulation_0031 / 0031_segmented) should get the same answer; disagreement marks errors.

  python experiments/error_map.py measure_v19 submission_v10
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402

SOL, SUB = sys.argv[1], sys.argv[2]


def base(vid: str) -> str:
    return re.sub(r"(_segmented|(?<=\d)[sx]|(?<=\d[a-z])[sx])$", "", vid)


def annotate(split):
    df = m2.load_split(split)
    cache = RESULTS / "cache" / split
    parsed = m2.load_parsed(df, cache)
    sol = pd.read_csv(cache / f"{SOL}_solutions.csv")[["id", "value", "reason"]]
    df = df.merge(sol, on="id", how="left")
    df["q"] = [parsed.get(i, {}).get("target_quantity") or "unparsed" for i in df.id]
    df["refq"] = [parsed.get(i, {}).get("ref_quantity") or "unparsed" for i in df.id]
    df["src"] = df.video_id.str.split("_").str[0]
    df["variant"] = np.where(df.video_id.str.endswith("_segmented") | df.video_id.str.match(r".*\d[a-z]?s$"), "seg",
                    np.where(df.video_id.str.match(r".*\d[a-z]?x$"), "x", "orig"))
    df["base"] = df.video_id.map(base)
    return df


def show(title, t):
    print(f"\n--- {title}"); print(t.to_string())


v = annotate("validation")
vp = pd.read_csv(RESULTS / "validation_predictions" / f"val_{SOL}.csv")
v = v.merge(vp.rename(columns={vp.columns[0]: "id"})[["id", "parsed_value", "source"]], on="id")
v["mra"] = mra(v.parsed_value, v.answer)
v["logerr"] = np.log(v.parsed_value / v.answer)
v["dir"] = np.where(v.logerr > np.log(1.25), "too big", np.where(v.logerr < -np.log(1.25), "too small", "ok"))
same = v.groupby(["base", "question"]).answer.agg(["nunique", "count"])
print(f"validation: variant groups with >1 clip: {(same['count'] > 1).sum()}, of which identical truth: "
      f"{((same['count'] > 1) & (same['nunique'] == 1)).sum()}")
agg = lambda g: pd.Series({"n": len(g), "mra": g.mra.mean(), "too_big": (g.dir == "too big").mean(),
                           "too_small": (g.dir == "too small").mean(), "lost": ((1 - g.mra).sum())})
show("validation by category x source", v.groupby(["category", "source"]).apply(agg).round(2))
show("validation by target quantity x source (lost = MRA points lost, summed)",
     v.groupby(["q", "source"]).apply(agg).round(2).sort_values("lost", ascending=False))
show("validation by video source x variant", v.groupby(["src", "variant"]).apply(agg).round(2))
show("validation by prior quantity (measured only)", v[v.source == "measure"].groupby("refq").apply(agg).round(2))

t = annotate("test")
tp = pd.read_csv(RESULTS / "submissions" / f"{SUB}_sources.csv")[["id", "parsed_value", "source"]]
t = t.merge(tp, on="id")
grp = t.groupby(["base", "question"])
spread = grp.parsed_value.agg(lambda s: s.max() / s.min() if len(s) > 1 else np.nan).rename("spread")
t = t.merge(spread.reset_index(), on=["base", "question"])
tv = t[t.spread.notna()].copy()
tv["inconsistent"] = tv.spread > 1.25
print(f"\ntest: {len(tv)} of {len(t)} items have a variant twin")
ta = lambda g: pd.Series({"n": len(g), "inconsistent": g.inconsistent.mean(), "median_spread": g.spread.median()})
show("test variant disagreement by category x source", tv.groupby(["category", "source"]).apply(ta).round(2))
show("test variant disagreement by target quantity (measured only)",
     tv[tv.source == "measure"].groupby("q").apply(ta).round(2).sort_values("n", ascending=False))
show("test variant disagreement by prior quantity (measured only)",
     tv[tv.source == "measure"].groupby("refq").apply(ta).round(2))
show("test variant disagreement by video source (measured only)",
     tv[tv.source == "measure"].groupby("src").apply(ta).round(2))
show("test: share answered by source, per category", pd.crosstab(t.category, t.source, normalize="index").round(2))
