"""Synthetic check of Track.accel_segment on the reference-motion patterns seen in QuantiPhy clips:
rest -> fall -> rest, two separate tosses, flight then bounce, plain constant acceleration, and
constant velocity (must be rejected). Positions in px with 0.7 px tracking noise."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.solver import SolveError, Track  # noqa: E402

FPS, G = 30.0, 900.0  # px/s^2
rng = np.random.default_rng(0)
t = np.arange(0, 3.0, 1 / FPS)


def track(y, x=None):
    x = np.zeros_like(y) if x is None else x
    pts = np.stack([x, y], 1) + rng.normal(0, 0.7, (len(t), 2))
    return Track(t, pts, FPS)


def fall(t0, t1, v0=0.0):
    """Position while falling in [t0, t1], resting before and after."""
    tc = np.clip(t, t0, t1) - t0
    return v0 * tc + 0.5 * G * tc ** 2


toss = lambda t0, dur: np.where((t >= t0) & (t <= t0 + dur), -(G * dur / 2) * (t - t0) + 0.5 * G * (t - t0) ** 2, 0)
cases = {
    "rest -> fall -> rest": (track(fall(1.0, 1.6)), G),
    "two tosses": (track(toss(0.4, 0.7) + toss(1.8, 0.7)), G),
    "flight then bounce": (track(np.where(t < 1.8, fall(0, 1.8, -600), fall(0, 1.8, -600)[int(1.8 * FPS)]
                                          - 300 * (t - 1.8))), G),
    "constant acceleration": (track(0.5 * 400 * t ** 2, 60 * t), 400.0),
}
ok = True
for name, (tr, truth) in cases.items():
    mag, rel, a, b = tr.accel_segment(0, 3)
    err = mag / truth - 1
    ok &= abs(err) < 0.05
    print(f"{name:24s} truth {truth:6.0f}  est {mag:7.1f}  err {err:+.1%}  phase {a:.2f}-{b:.2f}s  relSE {rel:.3f}")
try:
    Track(t, np.stack([200 * t, 50 * t], 1) + rng.normal(0, 0.7, (len(t), 2)), FPS).accel_segment(0, 3)
    print("constant velocity: NOT rejected")
    ok = False
except SolveError as e:
    print(f"constant velocity rejected: {e}")
print("PASS" if ok else "FAIL")
