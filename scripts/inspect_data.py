"""Profile the validation and test splits: columns, category mix, question templates."""
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
pd.set_option("display.width", 220, "display.max_colwidth", 120, "display.max_columns", 30)

val = pd.read_csv(DATA / "validation" / "validation_dataset.csv")
test = pd.read_parquet(DATA / "test" / "test_dataset.parquet")

for name, df in [("VALIDATION", val), ("TEST", test)]:
    df = df.copy()
    df["category"] = df["inference_type"].str[0] + df["video_type"].str[1]
    print(f"\n===== {name}: {len(df)} rows, {df['video_id'].nunique()} videos =====")
    print("columns:", list(df.columns))
    print(df.head(2).T)
    print("\ncategory counts:\n", df["category"].value_counts().to_string())
    print("\nsource x category:\n", pd.crosstab(df["video_source"], df["category"]).to_string())
    print("\ninference_type:\n", df["inference_type"].value_counts().to_string())
    # Question templates: mask object names roughly by keeping leading words and units
    unit = df["question"].str.extract(r"in ([^?]+)\?\s*$")[0]
    print("\nrequested units:\n", unit.value_counts().head(25).to_string())
    qtype = df["question"].str.extract(r"^What is the (\w+(?: \w+)?)", flags=re.I)[0].str.lower()
    print("\nquestion head words:\n", qtype.value_counts().head(25).to_string())

print("\n===== validation depth_info / prior samples =====")
v3 = val[val["video_type"].str[1] == "3"]
print(v3[["video_id", "inference_type", "question", val.columns[val.columns.str.contains("prior")][0], "depth_info"]].head(8).to_string())
print("\n===== prior-column / answer-column names =====")
print([c for c in val.columns if "prior" in c or "posterior" in c or "answer" in c])
