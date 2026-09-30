"""Print the solver's reason per item for given clips.  python scripts/show_reasons.py <split> <solutions> <video>..."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.scoring import RESULTS  # noqa: E402

split, name, vids = sys.argv[1], sys.argv[2], sys.argv[3:]
d = load_split(split).merge(pd.read_csv(RESULTS / "cache" / split / f"{name}_solutions.csv"), on="id")
pd.set_option("display.width", 220, "display.max_colwidth", 80)
print(d[d["video_id"].isin(vids)][["video_id", "question", "value", "reason"]].to_string(index=False))
