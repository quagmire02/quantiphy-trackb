"""Test-set composition by scoring category x target quantity (what share of the final score each cell carries)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.error_analysis import target_kind  # noqa: E402
from pipeline.data import load_split  # noqa: E402

for split in ("test", "validation"):
    df = load_split(split)
    df["target"] = df["question"].map(target_kind)
    ct = pd.crosstab(df["category"], df["target"])
    # Each category is 25% of the final score, split evenly across its items.
    weight = ct.div(ct.sum(axis=1), axis=0) * 25
    print(f"\n{split}: item counts")
    print(ct.to_string())
    print(f"{split}: % of final score")
    print(weight.round(1).to_string())
