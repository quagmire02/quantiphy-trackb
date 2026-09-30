"""Draw the motion regions found in clips (visual check).  python scripts/show_motion_regions.py <out.jpg> <video>..."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.measure_2d import TRACK_LONGEST  # noqa: E402
from pipeline.data import load_split  # noqa: E402
from pipeline.motion import motion_regions  # noqa: E402
from pipeline.video import read_frames  # noqa: E402

out, vids = sys.argv[1], sys.argv[2:]
paths = pd.concat([load_split("validation"), load_split("test")]).drop_duplicates("video_id").set_index("video_id")["video_path"]
tiles = []
for vid in vids:
    frames = read_frames(paths[vid], TRACK_LONGEST)
    regs = motion_regions(frames)
    img = frames[regs[0]["frame"] if regs else len(frames) // 2].copy()
    for k, r in enumerate(regs):
        x1, y1, x2, y2 = map(int, r["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"#{k} f{r['frame']}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(img, vid, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    tiles.append(cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1]))))
    print(vid, [(r["box"], r["frame"], r["area"]) for r in regs])
h = max(t.shape[0] for t in tiles)
tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
cv2.imwrite(out, cv2.cvtColor(np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
                              if len(tiles) % 2 == 0 else np.vstack(tiles), cv2.COLOR_RGB2BGR))
