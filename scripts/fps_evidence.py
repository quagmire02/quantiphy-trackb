"""Which frame rate is the true time base for videos whose file fps differs from the dataset's fps?
(a) videos present in both splits: validation copies (where file fps == dataset fps, and our answers
    score well) vs test copies -> clip durations under each time base;
(b) latest time mentioned in the questions vs the clip duration under each time base."""
import re
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402


def meta(path):
    cap = cv2.VideoCapture(path)
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # count decodable frames exactly (CAP_PROP_FRAME_COUNT can be an estimate)
    m = 0
    while cap.grab():
        m += 1
    cap.release()
    return fps, n, m


val, test = load_split("validation"), load_split("test")
tv = test.drop_duplicates("video_id").set_index("video_id")
rows = []
for vid, g in test.groupby("video_id"):
    fps_file, _, n = meta(g.video_path.iloc[0])
    csv = float(g.fps.iloc[0])
    if abs(fps_file / csv - 1) <= 0.02:
        continue
    times = [float(x) for q in g.question for x in re.findall(r"(\d+\.?\d*)\s*s\b", q)]
    rows.append({"vid": vid, "frames": n, "fps_csv": csv, "fps_file": round(fps_file, 2),
                 "dur_csv": round((n - 1) / csv, 2), "dur_file": round((n - 1) / fps_file, 2),
                 "max_t_asked": max(times) if times else None})
r = pd.DataFrame(rows)
print("(b) test videos with mismatched fps:", len(r))
asked = r.dropna(subset=["max_t_asked"])
print(f"    questions ask beyond the clip end: with dataset fps {int((asked.max_t_asked > asked.dur_csv + 0.05).sum())} "
      f"videos, with file fps {int((asked.max_t_asked > asked.dur_file + 0.05).sum())} (of {len(asked)} that ask a time)")
print(asked.sort_values("max_t_asked", ascending=False).head(12).to_string(index=False))

print("\n(a) same video in both splits:")
for vid, g in val.groupby("video_id"):
    if vid not in tv.index or vid not in set(r.vid):
        continue
    vf, _, vn = meta(g.video_path.iloc[0])
    t = r[r.vid == vid].iloc[0]
    print(f"  {vid}: validation {vn} frames @ {g.fps.iloc[0]} (file {vf:.2f}) = {(vn - 1) / g.fps.iloc[0]:.2f}s | "
          f"test {t.frames} frames: dataset fps {t.fps_csv} -> {t.dur_csv}s, file fps {t.fps_file} -> {t.dur_file}s")
