"""Sanity check: parse a few validation items and draw Qwen grounding boxes on their middle frames."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.ground import ground  # noqa: E402
from pipeline.parse import build_messages, finalize  # noqa: E402
from pipeline.video import read_frames, resize_longest  # noqa: E402
from pipeline.vlm import QwenVL  # noqa: E402

out_path = sys.argv[1]
df = load_split("validation")
picks = df.groupby("video_id").head(1).head(6)
vlm = QwenVL("Qwen/Qwen3-VL-4B-Instruct")
tiles = []
for row in picks.itertuples():
    rec = finalize(vlm.chat(build_messages(row.prior, row.question), max_new_tokens=160), row.prior, row.question)
    print(f"\n{row.video_id} | {row.prior} | {row.question}\n  -> {json.dumps({k: v for k, v in rec.items() if k != 'raw'})}")
    frames = read_frames(row.video_path)
    f = resize_longest(frames[len(frames) // 2][None], 1024, multiple=2)[0].copy()
    for obj, color in [(rec["ref_object"], (255, 0, 0))] + [(t, (0, 200, 0)) for t in rec["targets"]]:
        if not obj:
            continue
        g = ground(vlm, f, obj)
        print(f"  ground {obj!r}: {g['box']}  raw={g['reply'][:80]!r}")
        if g["box"]:
            x1, y1, x2, y2 = map(int, g["box"])
            cv2.rectangle(f, (x1, y1), (x2, y2), color, 3)
            cv2.putText(f, obj, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    tiles.append(cv2.resize(f, (512, int(512 * f.shape[0] / f.shape[1]))))
h = max(t.shape[0] for t in tiles)
tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
