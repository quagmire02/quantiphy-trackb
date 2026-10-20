"""Validation items whose answer changes with refined sizes: truth, before, after (per category)."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402

df = m2.load_split("validation"); cache = RESULTS / "cache/validation"
parsed = m2.load_parsed(df, cache)
res = {}
for on in (False, True):
    m2.REFINE_SIZES = on
    res[on] = m2.stage_solve(df, parsed, cache).set_index("id")
d = df.set_index("id")
a, b = res[False]["value"], res[True]["value"]
ch = (a - b).abs() > 1e-9 * a.abs()
t = pd.DataFrame({"cat": d.category, "vid": d.video_id, "q": d.question.str[:58], "truth": d.answer,
                  "before": a.round(3), "after": b.round(3), "mra_b": mra(a, d.answer), "mra_a": mra(b, d.answer),
                  "refq": [parsed[i]["ref_quantity"] for i in d.index]})[ch]
print(t.groupby("cat")[["mra_b", "mra_a"]].agg(["mean", "count"]).round(2).to_string())
print(t[t.cat == "D3"].sort_values("vid").to_string())
