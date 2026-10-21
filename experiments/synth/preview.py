"""Contact sheet of synthetic clips (first, middle, last frame) with their prior and questions."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

out = Path(sys.argv[1])
rows = [json.loads(line) for line in (out / "items.jsonl").read_text().splitlines()]
clips = list(dict.fromkeys(r["clip"] for r in rows))[: int(sys.argv[2]) if len(sys.argv) > 2 else 6]
sheet = []
for c in clips:
    cap = cv2.VideoCapture(str(out / "videos" / f"{c}.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    tiles = [cv2.resize(frames[i], (320, 180)) for i in (0, len(frames) // 2, len(frames) - 1)]
    sheet.append(np.hstack(tiles))
    items = [r for r in rows if r["clip"] == c]
    print(f"{c} ({items[0]['video_type']}, {len(frames)} frames @ {items[0]['fps']} fps, alpha {items[0]['alpha']}): prior: {items[0]['prior']}")
    if items[0]["depth_info"]:
        print("   depth:", items[0]["depth_info"].replace("\n", " | "))
    for r in items:
        print(f"   Q: {r['question']}  ->  {r['answer_text']}")
cv2.imwrite(str(out / "preview.jpg"), np.vstack(sheet))
