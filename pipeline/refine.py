"""Native-resolution mask refinement for object sizes.

Tracking runs on frames downscaled to 1024 px (longest side) and SAM 2 predicts masks on a coarse
grid, so the outline of a small object is only a few cells wide. For size measurement we re-cut the
object's mask on a few frames: crop around its tracked box from the video at native resolution
(test clips are 720p-4K) and segment the crop with the same SAM 2 (it rescales the crop to 1024 px,
i.e. zooms in). Extents are converted back to tracking-frame pixels, so the solvers are unchanged.

A refined mask is kept only if it stays inside the crop and its long side agrees with the tracked
one within a factor 1.6 (otherwise SAM segmented something else, e.g. a part or the background).
"""
import cv2
import numpy as np

from pipeline.geometry import F, mask_features

MAX_FRAMES = 6
MARGIN = 0.5          # crop margin, as a fraction of the box's long side
MIN_MARGIN_PX = 16    # ... and at least this many native pixels
MAX_TRACK_PX = 160    # refine objects up to this long side in tracking pixels (larger ones gain little)
AGREE = 1.6


def read_selected(path: str, wanted: set[int]) -> dict[int, np.ndarray]:
    """Native-resolution RGB frames for the wanted indices (decoded sequentially, others discarded)."""
    cap = cv2.VideoCapture(path)
    out, i, last = {}, 0, max(wanted) if wanted else -1
    while i <= last and cap.grab():
        if i in wanted:
            ok, fr = cap.retrieve()
            if ok:
                out[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return out


def pick_frames(f: np.ndarray, prefer: list[int]) -> list[int]:
    """Up to MAX_FRAMES frames where the object is present and fully inside the frame, spread over the
    clip, plus preferred frames (e.g. the 3D distance timestamps) when usable."""
    inner = np.flatnonzero((f[:, F["present"]] == 1) & (f[:, F["touches_border"]] == 0))
    if not len(inner):
        return []
    spread = inner[np.unique(np.linspace(0, len(inner) - 1, MAX_FRAMES).round().astype(int))].tolist()
    near = [int(inner[np.argmin(np.abs(inner - p))]) for p in prefer if np.min(np.abs(inner - p)) <= 2]
    return sorted(set(spread) | set(near))


def refine_object(tracker, native: dict[int, np.ndarray], f: np.ndarray, frames: list[int],
                  s_track: float) -> dict[int, list[float]]:
    """{frame: [width, height, major, minor, eq_diam]} in tracking-frame pixels."""
    out = {}
    for fi in frames:
        if fi not in native:
            continue
        img = native[fi]
        H, W = img.shape[:2]
        x1, y1, x2, y2 = (f[fi, [F["x1"], F["y1"], F["x2"], F["y2"]]] / s_track).tolist()
        m = max(MIN_MARGIN_PX, MARGIN * max(x2 - x1, y2 - y1))
        cx1, cy1 = int(max(0, x1 - m)), int(max(0, y1 - m))
        cx2, cy2 = int(min(W, x2 + m)), int(min(H, y2 + m))
        crop = np.ascontiguousarray(img[cy1:cy2, cx1:cx2])
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            continue
        box = [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]
        _, masks = tracker.track(crop[None], np.array([0]), {"o": (0, box)})
        g = mask_features(masks["o"][0])
        if g[F["present"]] != 1 or g[F["touches_border"]] == 1:
            continue
        ext = np.array([g[F["x2"]] - g[F["x1"]], g[F["y2"]] - g[F["y1"]], g[F["major"]], g[F["minor"]],
                        g[F["eq_diam"]]]) * s_track
        ratio = ext[2] / max(f[fi, F["major"]], 1e-6)
        if 1 / AGREE <= ratio <= AGREE:
            out[int(fi)] = [round(float(v), 3) for v in ext]
    return out


def apply_refined(feats: dict[str, np.ndarray], refined: dict[str, dict]) -> dict[str, np.ndarray]:
    """Copy of feats where sizes come from the refined frames only: their extents replace the tracked
    ones, and every other present frame is flagged as touching the border, which makes
    solver.size_series use only the refined frames. Positions (cx, cy) are untouched."""
    out = dict(feats)
    for key, per_frame in refined.items():
        if key not in feats or not per_frame:
            continue
        f = feats[key].copy()
        present = f[:, F["present"]] == 1
        f[present, F["touches_border"]] = 1
        for fi, (w, h, major, minor, eq) in per_frame.items():
            fi = int(fi)
            cx, cy = f[fi, F["cx"]], f[fi, F["cy"]]
            f[fi, [F["x1"], F["x2"]]] = cx - w / 2, cx + w / 2
            f[fi, [F["y1"], F["y2"]]] = cy - h / 2, cy + h / 2
            f[fi, [F["major"], F["minor"], F["eq_diam"]]] = major, minor, eq
            f[fi, F["touches_border"]] = 0
        out[key] = f
    return out
