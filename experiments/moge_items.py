"""Validation items whose 3D solve used a MoGe-transferred distance: measured vs VLM fallback vs truth."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "measure_v6"
cache = RESULTS / "cache" / "validation"
df = load_split("validation")
sol = pd.read_csv(cache / f"{name}_solutions.csv")
vlm = pd.read_csv(RESULTS / "validation_predictions" / "Qwen3-VL-4B_video_f16_px448.csv")[["id", "parsed_value"]]
d = df.merge(sol, on="id").merge(vlm.rename(columns={"parsed_value": "vlm"}), on="id")
m = d[d["moge_used"].astype(str) == "True"].copy()
m["ratio"] = (m["value"] / m["answer"]).round(2)
print(f"{len(m)} items used a MoGe-transferred distance: measured MRA {mra(m['value'], m['answer']).mean() * 100:.1f} "
      f"vs VLM MRA {mra(m['vlm'], m['answer']).mean() * 100:.1f} on the same items")
pd.set_option("display.width", 220, "display.max_colwidth", 70)
print(m[["video_id", "question", "answer", "value", "vlm", "ratio"]].round(3).to_string(index=False))
