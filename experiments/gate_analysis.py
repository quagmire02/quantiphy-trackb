"""Which cheap signals separate good measurements from catastrophic ones? (validation, 2D)

Signals per solved item:
  vlm_ratio      measured / VLM answer (disagreement with world-knowledge guess)
  ref_motion_px  reference displacement over the clip in px (tiny => speed prior unusable)
  tgt_area_frac  median target mask area / frame area (huge => segmented background, e.g. floor)
Then scores the S2/D2 MRA under simple fallback rules.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.measure_2d import load_parsed, object_keys, stage_solve  # noqa: E402
from pipeline import solver  # noqa: E402
from pipeline.data import load_split  # noqa: E402
from pipeline.geometry import F  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402

pd.set_option("display.width", 220)
cache = RESULTS / "cache" / "validation"
val = load_split("validation")
df = val[val["category"].isin(["S2", "D2"])]
parsed = load_parsed(df, cache)
# Re-solve with the reference-motion gate disabled so "no gate" really means no gate.
solver.SETTINGS["min_ref_motion_px"] = 0
sol = stage_solve(df, parsed, cache)
vlm = pd.read_csv(RESULTS / "validation_predictions" / "Qwen3-VL-4B_video_f16_px640.csv")[["id", "parsed_value"]]
d = df.merge(sol, on="id").merge(vlm.rename(columns={"parsed_value": "vlm"}), on="id")

sig = []
for row in d.itertuples():
    s = {"id": row.id, "ref_motion_px": np.nan, "tgt_area_frac": np.nan}
    rec = parsed.get(row.id)
    tpath = cache / "tracks" / f"{row.video_id}.npz"
    if rec and rec["ok"] and tpath.exists() and np.isfinite(row.value):
        feats = dict(np.load(tpath))
        shape = json.loads((cache / "ground" / f"{row.video_id}.json").read_text())["shape"]
        ref, tgts = object_keys(rec)
        r = feats[ref][feats[ref][:, F["present"]] == 1]
        s["ref_motion_px"] = float(np.hypot(*(r[:, [F["cx"], F["cy"]]].max(0) - r[:, [F["cx"], F["cy"]]].min(0))))
        t = feats[tgts[0]]
        s["tgt_area_frac"] = float(np.nanmedian(t[t[:, F["present"]] == 1][:, F["area"]]) / (shape[0] * shape[1]))
    sig.append(s)
d = d.merge(pd.DataFrame(sig), on="id")
d["vlm_ratio"] = d["value"] / d["vlm"]
d["err"] = d["value"] / d["answer"]
d["mra_meas"] = mra(d["value"], d["answer"])
d["mra_vlm"] = mra(d["vlm"], d["answer"])

solved = d[d["value"].notna()]
print(solved[["video_id", "category", "err", "vlm_ratio",
              "ref_motion_px", "tgt_area_frac"]].round(3).sort_values("err").to_string(index=False))


def score(use_meas):
    pred = np.where(use_meas & d["value"].notna(), d["value"], d["vlm"])
    per = mra(pred, d["answer"])
    return {c: per[(d["category"] == c).values].mean() * 100 for c in ["S2", "D2"]}


print("\nFallback rules (S2 / D2 MRA):")
is_speed_prior = d["inference_type"].str[0] == "D"
rules = {
    "no gate": np.ones(len(d), bool),
    "vlm disagreement <= 10x": np.abs(np.log10(d["vlm_ratio"])) <= 1,
    "vlm disagreement <= 5x": np.abs(np.log10(d["vlm_ratio"])) <= np.log10(5),
    "target area <= 20% of frame": ~(d["tgt_area_frac"] > 0.2),
    "speed prior: ref moves >= 20px": ~(is_speed_prior & (d["ref_motion_px"] < 20)),
}
rules["area + ref motion"] = rules["target area <= 20% of frame"] & rules["speed prior: ref moves >= 20px"]
rules["area + ref motion + vlm<=10x"] = rules["area + ref motion"] & rules["vlm disagreement <= 10x"]
for name, m in rules.items():
    s = score(m.values if hasattr(m, "values") else m)
    print(f"  {name:32s} S2 {s['S2']:5.1f}  D2 {s['D2']:5.1f}  kept {int((m & d['value'].notna()).sum())}")
