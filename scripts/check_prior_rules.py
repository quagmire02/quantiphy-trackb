"""Print rule-based prior parsing for every distinct prior in validation + test (review by eye)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.parse import first_prior, prior_description, prior_object, prior_quantity, prior_time  # noqa: E402
from pipeline.units import question_unit  # noqa: E402
from pipeline.units import parse_prior_value  # noqa: E402

priors = pd.concat([load_split("validation"), load_split("test")])["prior"].astype(str).str.strip().unique()
rows = []
for p in sorted(priors):
    d = prior_description(p)
    rows.append({"prior": p, "object": prior_object(d), "quantity": prior_quantity(d), "time": prior_time(d),
                 "value_unit": parse_prior_value(first_prior(p))})
df = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 500)
show = df[df["prior"].str.contains(r"speedboat|from \d|gravity_|lenth|dollar|\n|radius|ruler", regex=True)
          | (df["object"] == "") | df["quantity"].isna()]
print(show.to_string(index=False))
questions = pd.concat([load_split("validation"), load_split("test")])["question"].unique()
units = pd.Series([question_unit(q) for q in questions])
print("\nquestion units:", units.value_counts(dropna=False).to_dict())
print("no unit, sample:", [q for q, u in zip(questions, units) if u is None][:8])
print(f"\n{len(df)} priors; no object: {(df['object'] == '').sum()}; no quantity: {df['quantity'].isna().sum()}; "
      f"with time: {df['time'].notna().sum()}; no value: {df['value_unit'].map(lambda v: v[0] is None).sum()}")
