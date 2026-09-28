"""Dump prior phrasings and a sample of question phrasings (for designing the parser)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402

df = pd.concat([load_split("test"), load_split("validation")])
priors = df.drop_duplicates("video_id")["prior"]
print(f"{priors.nunique()} distinct priors over {df['video_id'].nunique()} videos")
print("\n".join(priors.sample(60, random_state=0).astype(str)))
print("\nPRIORS WITH TIME:", priors[priors.str.contains(r"\d\s*s\b|t\s*=|at ", regex=True)].head(15).tolist())
print("\nPRIORS WITHOUT '=':", priors[~priors.str.contains("=")].head(15).tolist())
print("\nQUESTIONS (2D sample):")
q2 = df[df["category"].str[1] == "2"]["question"]
print("\n".join(q2.sample(60, random_state=1)))
print("\nDEPTH_INFO sample:", df["depth_info"].dropna().drop_duplicates().sample(4, random_state=0).tolist())
