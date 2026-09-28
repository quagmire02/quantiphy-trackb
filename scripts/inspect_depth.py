"""Look at depth_info in 3D items: formats, timestamps, and whether prior / question objects are covered."""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.parse import prior_description, prior_object  # noqa: E402

LINE = re.compile(r"(?:t\s*=\s*([\d.]+)\s*s?\s*,\s*)?distance_(.+?)_camera\s*=\s*([\d.]+)\s*(m|cm)?", re.I)

for split in ("validation", "test"):
    df = load_split(split)
    d3 = df[df["video_type"].str[1] == "3"]
    vids = d3.drop_duplicates("video_id")
    print(f"\n===== {split}: {len(d3)} 3D items, {len(vids)} videos")
    n_lines = n_parsed = n_timed = ref_cov = 0
    unparsed = []
    for row in vids.itertuples():
        lines = [l.strip() for l in str(row.depth_info).splitlines() if l.strip()]
        names = set()
        for l in lines:
            n_lines += 1
            m = LINE.search(l)
            if m:
                n_parsed += 1
                n_timed += m.group(1) is not None
                names.add(m.group(2).replace("_", " ").lower())
            else:
                unparsed.append(l)
        ref = prior_object(prior_description(row.prior))
        if ref and any(set(ref.split()) & set(n.split()) for n in names):
            ref_cov += 1
    print(f"depth lines {n_lines}, parsed {n_parsed}, with time {n_timed}; videos whose prior object has a depth entry: "
          f"{ref_cov}/{len(vids)}")
    print("unparsed lines:", unparsed[:10])
    if split == "validation":
        for row in vids.itertuples():
            qs = d3[d3["video_id"] == row.video_id]["question"].tolist()
            print(f"\n{row.video_id} [{row.category}] prior: {row.prior}\n  depth: {str(row.depth_info)!r}")
            for q in qs[:6]:
                print("   Q:", q)
