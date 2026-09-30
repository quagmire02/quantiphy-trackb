"""Why does accel_segment reject a clip? Fit statistics of the whole prior window and its best sub-runs.
  python scripts/debug_segment.py <split> <video> <solutions>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.measure_2d import load_parsed, object_keys  # noqa: E402
from pipeline.data import load_split  # noqa: E402
from pipeline.scoring import RESULTS  # noqa: E402
from pipeline.solver import SETTINGS, Track, _resolve  # noqa: E402

split, vid = sys.argv[1], sys.argv[2]
cache = RESULTS / "cache" / split
df = load_split(split)
items = df[df["video_id"] == vid]
parsed = load_parsed(items, cache)
rec = next(parsed[i] for i in items["id"] if parsed.get(i) and parsed[i]["ok"])
ref = object_keys(rec)[0]
f = dict(np.load(cache / "tracks" / f"{vid}.npz"))[ref]
fps = float(items["fps"].iloc[0])
tr = Track.from_feats(f, fps)
n = len(f)
tw = _resolve(rec["ref_time"], (n - 1) / fps)
a, b = tw if tw else (0.0, (n - 1) / fps)
sel = (tr.t >= a - 1e-6) & (tr.t <= b + 1e-6)
t, p = tr.t[sel], tr.pts[sel]
print(f"{vid}: fps {fps}, prior window [{a:.2f},{b:.2f}], {sel.sum()} tracked points, times {np.round(t, 3)}")
d2 = np.diff(p, 2, axis=0)
sigma = float(np.max(1.4826 * np.median(np.abs(d2 - np.median(d2, 0)), 0) / np.sqrt(6)))
print(f"noise sigma {sigma:.2f}px -> tol {max(SETTINGS['segment_max_resid_px'], SETTINGS['segment_noise_k'] * sigma):.2f}px")
print("y:", np.round(p[:, 1] - p[0, 1], 1))
try:
    print("accel_segment ->", tr.accel_segment(a, b))
except Exception as e:
    print("accel_segment ->", type(e).__name__, e)
print("robust fit, run 0-22:", Track._robust_parabola(t[:22], p[:22], 6.03))
tol_px = max(SETTINGS["segment_max_resid_px"], SETTINGS["segment_noise_k"] * sigma)
for length in range(len(t), 19, -1):
    tt, pp = t[:length], p[:length]
    tol = max(tol_px, SETTINGS["segment_resid_frac"] * float(np.linalg.norm(pp.max(0) - pp.min(0))))
    print(f"  length {length}: tol {tol:.2f} robust fit {Track._robust_parabola(tt, pp, tol)}")
for lo, hi in [(0, len(t))] + [(i, j) for i in range(0, 4) for j in range(len(t) - 3, len(t) + 1) if j - i >= 5][:12]:
    tt, pp = t[lo:hi], p[lo:hi]
    tc = tt - tt.mean()
    acc, var, res = [], [], []
    for k in range(2):
        coef, cov = np.polyfit(tc, pp[:, k], 2, cov=True)
        acc.append(2 * coef[0]); var.append(4 * cov[0, 0]); res.append(pp[:, k] - np.polyval(coef, tc))
    acc, var = np.array(acc), np.array(var)
    mag = np.linalg.norm(acc)
    rel = np.sqrt((acc ** 2 * var).sum()) / mag ** 2
    print(f"  run {lo:2d}-{hi:2d} [{tt[0]:.3f}-{tt[-1]:.3f}]  a={mag:8.1f}px/s2  relSE {rel:.3f}  "
          f"max|res| {np.abs(np.array(res)).max():.2f}px")
