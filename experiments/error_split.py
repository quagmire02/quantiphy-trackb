"""Where measured error comes from (validation): clip-level scale (shared by all items of a clip,
i.e. the reference / calibration) vs item-level (each target's own measurement); and how much
the VLM fallback items cost."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split
from pipeline.scoring import RESULTS, mra

df = load_split("validation")
p = pd.read_csv(RESULTS / f"validation_predictions/val_{sys.argv[1]}.csv")
df = df.merge(p.rename(columns={p.columns[0]: "id"})[["id", "parsed_value", "source"]], on="id")
df["mra"] = mra(df.parsed_value, df.answer)
df["logerr"] = np.log(df.parsed_value / df.answer)
m = df[df.source == "measure"].copy()
m["clip_bias"] = m.groupby("video_id").logerr.transform("median")
m["resid"] = m.logerr - m.clip_bias
print(f"measured items {len(m)} in {m.video_id.nunique()} clips")
print(f"  total log-error variance {m.logerr.var():.3f} = clip-level {m.clip_bias.var():.3f} + item-level {m.resid.var():.3f}")
# what if the clip scale were perfect (remove each clip's shared bias)?
m["fixed"] = m.answer * np.exp(m.resid)
print(f"  measured MRA now {m.mra.mean():.3f}; with a perfect per-clip scale {mra(m.fixed, m.answer).mean():.3f}")
for c, g in m.groupby("category"):
    print(f"   {c}: now {g.mra.mean():.3f}  perfect scale {mra(g.fixed, g.answer).mean():.3f}  (n={len(g)})")
v = df[df.source != "measure"]
print(f"fallback items {len(v)} ({len(v)/len(df):.0%}), MRA {v.mra.mean():.3f}; "
      f"share of all lost MRA: {(1 - v.mra).sum() / (1 - df.mra).sum():.0%}")
print("per category: fallback share / fallback MRA / measured MRA")
for c, g in df.groupby("category"):
    f = g[g.source != "measure"]; mm = g[g.source == "measure"]
    print(f"   {c}: {len(f)/len(g):.0%} / {f.mra.mean():.2f} / {mm.mra.mean():.2f}")
