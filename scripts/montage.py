"""Save a first/middle/last-frame strip for each given video id (visual debugging).

  python scripts/montage.py OUT.jpg captured_0027s simulation_0170_segmented ...
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.video import read_frames  # noqa: E402

out, ids = sys.argv[1], sys.argv[2:]
import pandas as pd  # noqa: E402

paths = pd.concat([load_split("validation"), load_split("test")]).drop_duplicates("video_id").set_index("video_id")["video_path"]
rows = []
for vid in ids:
    fr = read_frames(paths[vid])
    strip = []
    for i in (0, len(fr) // 2, len(fr) - 1):
        f = cv2.resize(fr[i], (400, int(400 * fr.shape[1] / fr.shape[2])))
        f = cv2.resize(f, (400, 225))
        cv2.putText(f, f"{vid} f{i}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        strip.append(f)
    rows.append(np.hstack(strip))
cv2.imwrite(out, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
print("wrote", out)
