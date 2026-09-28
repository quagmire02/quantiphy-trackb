"""How do the official submission template's ids map onto the test parquet rows?"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import DATA, load_split  # noqa: E402

tpl = pd.read_csv(DATA / "template" / "quantiphy_submission_template.csv")
test = load_split("test")
print("template:", tpl.shape, "columns:", list(tpl.columns))
print("id range:", tpl["id"].min(), "-", tpl["id"].max(), "unique:", tpl["id"].is_unique,
      "| parsed_value non-null:", tpl["parsed_value"].notna().sum())
print("test parquet rows:", len(test))

norm = lambda s: s.astype(str).str.strip()
same_order = (len(tpl) == len(test) and (norm(tpl["video_id"]).values == norm(test["video_id"]).values).all()
              and (norm(tpl["question"]).values == norm(test["question"]).values).all())
print("template rows in same order as parquet (video_id + question):", same_order)
if not same_order:
    key = lambda d: norm(d["video_id"]) + " || " + norm(d["question"])
    k_t, k_p = key(tpl), key(test)
    print("keys unique in template / parquet:", k_t.is_unique, k_p.is_unique)
    print("template keys missing from parquet:", (~k_t.isin(set(k_p))).sum(),
          "| parquet keys missing from template:", (~k_p.isin(set(k_t))).sum())
    diff = [(i, a, b) for i, (a, b) in enumerate(zip(k_t, k_p)) if a != b][:5]
    print("first differing rows:", diff)
print("\nid == parquet row index + 1 for all rows:", same_order and (tpl["id"].values == test["id"].values + 1).all())
