"""Video decoding helpers (OpenCV). Frame i is at time i / fps, with fps taken from the dataset CSV."""
import cv2
import numpy as np


def read_frames(path: str, longest: int | None = None) -> np.ndarray:
    """All frames as uint8 RGB array (T, H, W, 3); optionally downscaled while decoding
    (4K clips would otherwise need >1 GB of RAM)."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if longest and max(frame.shape[:2]) > longest:
            s = longest / max(frame.shape[:2])
            frame = cv2.resize(frame, (round(frame.shape[1] * s), round(frame.shape[0] * s)),
                               interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise IOError(f"could not decode {path}")
    return np.stack(frames)


def sample_indices(total: int, max_frames: int) -> np.ndarray:
    """Uniform indices covering first..last frame; even count (Qwen merges frame pairs)."""
    n = min(total, max_frames)
    n -= n % 2
    return np.unique(np.linspace(0, total - 1, max(n, 2)).round().astype(int))


def resize_longest(frames: np.ndarray, longest: int, multiple: int = 32) -> np.ndarray:
    """Resize so the longest side is <= `longest`, both sides snapped to `multiple` (Qwen3-VL patch*merge)."""
    h, w = frames.shape[1:3]
    s = min(1.0, longest / max(h, w))
    nh, nw = (max(multiple, int(round(d * s / multiple)) * multiple) for d in (h, w))
    if (nh, nw) == (h, w):
        return frames
    return np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in frames])
