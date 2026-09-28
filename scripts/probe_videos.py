"""Print resolution / frame count / fps for a sample of validation and test videos."""
from pathlib import Path

import cv2

DATA = Path(__file__).resolve().parents[1] / "data"
val_videos = sorted((DATA / "validation" / "validation_videos").glob("*"))
test_videos = sorted((DATA / "test").glob("*.mp4"))
print("validation video files:", [p.name for p in val_videos[:6]], "...")

for p in val_videos[:4] + test_videos[::80]:
    cap = cv2.VideoCapture(str(p))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"{p.parent.name:18s} {p.name:24s} {w}x{h}  {n} frames @ {fps:.2f} fps = {n / max(fps, 1e-6):.2f}s")
