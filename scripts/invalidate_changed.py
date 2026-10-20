"""After the 2026-10-09 dataset update: drop cached parse / grounding / tracking results of clips whose
questions or categories changed, so the pipeline recomputes them. Backs up what it removes.

Test: changed rows = upload template (still the old wording) vs the new parquet, plus the clips whose
video_type changed. Validation: the clips with corrected question wording.

  python scripts/invalidate_changed.py            # dry run
  python scripts/invalidate_changed.py --apply
"""
import json
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import DATA, load_split  # noqa: E402
from pipeline.scoring import RESULTS  # noqa: E402

APPLY = "--apply" in sys.argv
norm = lambda s: s.map(lambda v: re.sub(r"\s+", " ", str(v)).strip())  # noqa: E731

test = load_split("test")
tpl = pd.read_csv(DATA / "template" / "quantiphy_submission_template.csv")
changed = test[(norm(tpl["question"]) != norm(test["question"])).values]
clips = {"test": set(changed.video_id) | {"simulation_0022", "simulation_0247_segmented"},
         "validation": {"captured_0013", "captured_0041x", "internet_0013", "internet_0027"}}
stamp = time.strftime("%Y%m%d_%H%M%S")
for split, vids in clips.items():
    df = load_split(split)
    cache = RESULTS / "cache" / split
    ids = set(df.loc[df.video_id.isin(vids), "id"])
    print(f"{split}: {len(vids)} clips, {len(ids)} questions: {sorted(vids)}")
    if not APPLY:
        continue
    backup = cache / f"_invalidated_{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    for name, key in [("parse_question.jsonl", "id"), ("parse_prior.jsonl", "id")]:
        p = cache / name
        if not p.exists():
            continue
        shutil.copy(p, backup / name)
        recs = [json.loads(line) for line in p.read_text().splitlines()]
        drop = ids if name == "parse_question.jsonl" else vids
        keep = [r for r in recs if r[key] not in drop]
        p.write_text("".join(json.dumps(r) + "\n" for r in keep))
        print(f"   {name}: removed {len(recs) - len(keep)} entries")
    n = 0
    for vid in vids:
        for sub, ext in [("ground", "json"), ("tracks", "npz"), ("moge", "json"), ("refine", "json"), ("overlays", "jpg")]:
            f = cache / sub / f"{vid}.{ext}"
            if f.exists():
                (backup / sub).mkdir(exist_ok=True)
                shutil.move(str(f), backup / sub / f.name)
                n += 1
    print(f"   moved {n} cached files to {backup.name}")
