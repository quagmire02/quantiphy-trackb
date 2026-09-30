"""Synthetic check of the 3D solver: known camera + known motion -> does it recover the truth?"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.geometry import F, FEATURES  # noqa: E402
from pipeline.solver3d import solve_3d  # noqa: E402

W, H, F_TRUE, FPS, T = 1024, 576, 900.0, 30.0, 61
t = np.arange(T) / FPS


def render(P, radius):
    """Pixel features of a sphere at 3D positions P (T,3), no noise."""
    feats = np.full((T, len(FEATURES)), np.nan)
    u = W / 2 + F_TRUE * P[:, 0] / P[:, 2]
    v = H / 2 + F_TRUE * P[:, 1] / P[:, 2]
    r_px = F_TRUE * radius / P[:, 2]
    feats[:, F["present"]] = 1
    feats[:, F["cx"]], feats[:, F["cy"]] = u, v
    feats[:, F["area"]] = np.pi * r_px ** 2
    feats[:, F["eq_diam"]] = 2 * r_px
    feats[:, F["x1"]], feats[:, F["x2"]] = u - r_px, u + r_px
    feats[:, F["y1"]], feats[:, F["y2"]] = v - r_px, v + r_px
    feats[:, F["major"]] = feats[:, F["minor"]] = 2 * r_px
    feats[:, F["touches_border"]] = 0
    return feats


# Ball A (diameter 0.22 m) moves diagonally away from the camera at constant velocity;
# ball B (diameter 0.30 m) is static off-centre.
vA = np.array([1.2, -0.3, 2.0])
PA = np.array([-1.0, 0.2, 6.0]) + t[:, None] * vA
PB = np.tile([1.5, 0.4, 8.0], (T, 1))
fa, fb = render(PA, 0.11), render(PB, 0.15)
dist = lambda P, i: float(np.linalg.norm(P[i]))  # Euclidean camera distance
depth = (f"t=0.5s, distance_ball_a_camera = {dist(PA, 15):.4f} m\n"
         f"t=1.5s, distance_ball_a_camera = {dist(PA, 45):.4f} m\n"
         f"distance_ball_b_camera = {dist(PB, 0):.4f} m")

base = {"prior_value": 22.0, "prior_unit": "cm", "ref_quantity": "diameter", "ref_time": None}
cases = [
    ("diameter of ball B (m)", {"target_quantity": "diameter", "target_time": None, "target_unit": "m"},
     [fb], ["ball b"], 0.30),
    ("speed of ball A at 1.0s (m/s)", {"target_quantity": "speed", "target_time": [1.0, 1.0], "target_unit": "m/s"},
     [fa], ["ball a"], float(np.linalg.norm(vA))),
    ("distance A-B at 1.0s (m)", {"target_quantity": "distance", "target_time": [1.0, 1.0], "target_unit": "m"},
     [fa, fb], ["ball a", "ball b"], float(np.linalg.norm(PA[30] - PB[30]))),
    ("displacement A 0.5-1.5s (cm)", {"target_quantity": "displacement", "target_time": [0.5, 1.5], "target_unit": "cm"},
     [fa], ["ball a"], 100 * float(np.linalg.norm(PA[45] - PA[15]))),
]
ok = True
all_feats = {"ball a": fa, "ball b": fb}
for name, q, feats, keys, truth in cases:
    out = solve_3d({**base, **q}, all_feats, ("ball a", keys), depth, FPS, T, W, H)
    err = out["value"] / truth - 1
    ok &= abs(err) < 0.02
    print(f"{name:32s} truth {truth:8.4f}  solved {out['value']:8.4f}  err {err:+.2%}  f {out['f_px']:.1f} (true {F_TRUE})")

# Speed prior instead of size prior: calibrate f from ball A's average speed.
spd = {"prior_value": float(np.linalg.norm(vA)), "prior_unit": "m/s", "ref_quantity": "speed", "ref_time": None,
       "target_quantity": "diameter", "target_time": None, "target_unit": "m"}
out = solve_3d(spd, all_feats, ("ball a", ["ball b"]), depth, FPS, T, W, H)
err = out["value"] / 0.30 - 1
ok &= abs(err) < 0.02
print(f"{'speed prior -> diameter of B':32s} truth   0.3000  solved {out['value']:8.4f}  err {err:+.2%}  f {out['f_px']:.1f}")

# Ball C (diameter 0.40 m) has NO stated distance; MoGe-like distances with an arbitrary global scale
# (x0.6, as MoGe's metric scale is unreliable) transfer ball B's stated distance to it via the ratio.
PC = np.tile([-0.8, 0.3, 10.0], (T, 1))
fc = render(PC, 0.20)
moge = {fi: {"ball a": 0.6 * dist(PA, fi), "ball b": 0.6 * dist(PB, fi), "ball c": 0.6 * dist(PC, fi)}
        for fi in (15, 30, 45)}
feats_c = {**all_feats, "ball c": fc}
q = {"target_quantity": "diameter", "target_time": None, "target_unit": "m"}
out = solve_3d({**base, **q}, feats_c, ("ball a", ["ball c"]), depth, FPS, T, W, H, moge=moge)
err = out["value"] / 0.40 - 1
ok &= abs(err) < 0.02 and out["moge_used"]
print(f"{'C without distance (MoGe ratio)':32s} truth   0.4000  solved {out['value']:8.4f}  err {err:+.2%}  "
      f"moge_used {out['moge_used']}")
print("PASS" if ok else "FAIL")
