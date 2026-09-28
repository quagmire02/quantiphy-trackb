"""Summarise a raw VLM run: latency distribution and unparseable replies."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for name in sys.argv[1:]:
    recs = pd.DataFrame([json.loads(l) for l in (ROOT / "results" / "validation_raw" / f"{name}.jsonl").read_text().splitlines()])
    print(f"\n== {name}")
    print("seconds:", recs["seconds"].describe()[["mean", "50%", "max"]].round(1).to_dict())
    print("slowest videos:", recs.groupby("video_id")["seconds"].mean().sort_values().tail(4).round(1).to_dict())
    bad = recs[recs["value"].isna() | (recs["value"] == 0)]
    print(f"invalid {len(bad)}:", bad["reply"].value_counts().head(6).to_dict())
