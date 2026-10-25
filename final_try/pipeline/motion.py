"""Motion-guided re-grounding for references that must move.

A prior that gives an object's speed or acceleration implies the object moves. When the grounded
reference turns out static (e.g. 'moving chess piece' boxed on a picture), the grounding picked the
wrong instance. Here: find the regions that actually change over the clip (frame differencing),
ask the VLM for the object inside each region, and keep the answer that overlaps real motion.
"""
import cv2
import numpy as np


def motion_regions(frames: np.ndarray, samples: int = 24, max_regions: int = 4, diff_thresh: int = 18,
                   min_area: int = 30) -> list[dict]:
    """Regions of the clip that change between sampled frames, largest first:
    [{'box': [x1,y1,x2,y2], 'frame': frame index where that region changes most, 'area': px}]."""
    T = len(frames)
    idx = np.unique(np.linspace(0, T - 1, min(T, samples)).round().astype(int))
    gray = [cv2.GaussianBlur(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY), (5, 5), 0) for i in idx]
    diffs = [cv2.absdiff(a, b) > diff_thresh for a, b in zip(gray[:-1], gray[1:])]
    if not diffs:
        return []
    acc = np.sum(diffs, axis=0)
    mask = (acc >= 2).astype(np.uint8)  # changed in at least two intervals: real motion, not flicker
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    regions = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < min_area:
            continue
        comp = labels == k
        per_interval = [int((d & comp).sum()) for d in diffs]
        regions.append({"box": [float(x), float(y), float(x + w), float(y + h)], "area": int(area),
                        "frame": int(idx[int(np.argmax(per_interval))])})
    return sorted(regions, key=lambda r: -r["area"])[:max_regions]


def expand(box, W: int, H: int, factor: float = 1.8, min_side: int = 160):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max((x2 - x1) * factor, min_side), max((y2 - y1) * factor, min_side)
    return [max(0, cx - w / 2), max(0, cy - h / 2), min(W, cx + w / 2), min(H, cy + h / 2)]


def overlap_fraction(box, region) -> float:
    """Share of `box` that lies inside `region`."""
    ix = max(0.0, min(box[2], region[2]) - max(box[0], region[0]))
    iy = max(0.0, min(box[3], region[3]) - max(box[1], region[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return ix * iy / area if area > 0 else 0.0


def ground_moving(vlm, frames: np.ndarray, phrase: str, ground_fn) -> dict | None:
    """Box for `phrase` inside a moving region: {'frame', 'box', 'region'} or None."""
    H, W = frames.shape[1:3]
    for reg in motion_regions(frames):
        crop_box = [int(v) for v in expand(reg["box"], W, H)]
        x1, y1, x2, y2 = crop_box
        crop = np.ascontiguousarray(frames[reg["frame"], y1:y2, x1:x2])
        if crop.shape[0] < 16 or crop.shape[1] < 16:
            continue
        for b in ground_fn(vlm, crop, phrase)["boxes"]:
            box = [b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1]
            if overlap_fraction(box, expand(reg["box"], W, H, factor=1.2, min_side=0)) >= 0.5:
                return {"frame": reg["frame"], "box": [round(v, 1) for v in box], "region": reg["box"]}
    return None
