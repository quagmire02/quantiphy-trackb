"""Synthetic check of colour-word disambiguation: coloured boxes on a grey background."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.ground import colour_word, pick_by_colour  # noqa: E402

RGB = {"red": (200, 30, 30), "brown": (120, 70, 30), "white": (240, 240, 240), "yellow": (230, 200, 40),
       "black": (20, 20, 20), "blue": (30, 60, 200), "green": (40, 160, 60), "orange": (240, 130, 20),
       "grey": (128, 128, 128), "pink": (240, 150, 190), "purple": (120, 40, 170)}
frame = np.full((200, 1200, 3), 100, np.uint8)
boxes = []
for i, (name, rgb) in enumerate(RGB.items()):
    x = 10 + i * 105
    cv2.rectangle(frame, (x, 50), (x + 90, 150), rgb, -1)
    boxes.append([x, 50, x + 90, 150])
ok = True
for i, name in enumerate(RGB):
    others = [boxes[j] for j in range(len(boxes)) if j != i][:3]
    cands = [boxes[i]] + others
    pick = pick_by_colour(f"the {name} object", cands, frame)
    hit = pick is boxes[i]
    ok &= hit
    print(f"{name:7s} word={colour_word('the ' + name + ' object')!s:7s} picked correct: {hit}")
print("PASS" if ok else "FAIL")
