"""Which clips / items did each guard reject? (solutions CSV name as argument)"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.measure_2d import load_parsed  # noqa: E402
from pipeline.data import load_split  # noqa: E402
from pipeline.scoring import RESULTS  # noqa: E402

split, name = sys.argv[1], sys.argv[2]
df = load_split(split)
parsed = load_parsed(df, RESULTS / "cache" / split)
sol = pd.read_csv(RESULTS / "cache" / split / f"{name}.csv")
hits = sol[sol["reason"].astype(str).str.startswith("guard")].merge(df[["id", "video_id", "question", "prior"]], on="id")
hits["ref"] = hits["id"].map(lambda i: parsed[i]["ref_object"])
hits["targets"] = hits["id"].map(lambda i: parsed[i]["targets"])
pd.set_option("display.width", 250, "display.max_colwidth", 60, "display.max_rows", 300)
print(hits.groupby(["video_id", "prior", "reason"]).size().to_string())
print(hits[hits["reason"].str.contains("free fall")][["video_id", "ref", "targets", "question"]].to_string(index=False))
