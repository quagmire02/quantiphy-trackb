"""Target acceleration over the constant-acceleration phase vs local fit (validation labels, test changes)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2  # noqa: E402
import pipeline.solver as s2  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402

for split in ["validation", "test"]:
    df = m2.load_split(split); cache = RESULTS / "cache" / split
    parsed = m2.load_parsed(df, cache)
    res = {}
    for on in [False, True]:
        s2.SETTINGS["target_accel_segment"] = on
        res[on] = m2.stage_solve(df, parsed, cache).set_index("id")["value"]
    s2.SETTINGS["target_accel_segment"] = False
    d = df.set_index("id")
    ch = ~np.isclose(res[False].fillna(-1), res[True].fillna(-1))
    print(f"\n{split}: {ch.sum()} items change; by category {d.category[ch].value_counts().to_dict()}")
    if split == "validation":
        vlm = pd.read_csv(RESULTS / "validation_predictions/Qwen3-VL-4B_video_f16_px448.csv")
        vlm = vlm.set_index(vlm.columns[0])["parsed_value"]
        for on, v in res.items():
            full = v.fillna(vlm.reindex(v.index))
            print("segment" if on else "local  ", {c: round(mra(full[d.category == c], d.answer[d.category == c]).mean() * 100, 1)
                                                   for c in ["S2", "D2", "S3", "D3"]})
        print(pd.DataFrame({"vid": d.video_id[ch], "q": d.question[ch].str[:55], "ans": d.answer[ch],
                            "local": res[False][ch].round(3), "segment": res[True][ch].round(3)}).to_string())
    else:
        r = (res[True][ch] / res[False][ch]).dropna()
        print("ratio segment/local quantiles:", r.quantile([.1, .25, .5, .75, .9]).round(2).to_dict())
