"""First / middle / last frame of given synthetic clips, upscaled, side by side, with their text printed."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

data, out, clips = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
rows = [json.loads(line) for line in (data / "items.jsonl").read_text().splitlines()]
sheet = []
for c in clips:
    cap = cv2.VideoCapture(str(data / "videos" / f"{c}.mp4"))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    sheet.append(np.hstack([cv2.resize(fr[i], (426, 240)) for i in (0, len(fr) // 2, len(fr) - 1)]))
    items = [r for r in rows if r["clip"] == c]
    print(c, "|", items[0]["prior"], "|", "; ".join(f"{r['question']} -> {r['answer_text']}" for r in items))
cv2.imwrite(out, np.vstack(sheet))
