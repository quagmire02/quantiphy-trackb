"""2D solver: tracked pixel geometry + one metric prior -> requested physical quantity.

Assumes planar motion at roughly constant depth (the benchmark's 2D setting), so a single
metres-per-pixel scale holds for the whole frame and clip.
"""
import numpy as np
from scipy.signal import savgol_filter

from pipeline.geometry import F
from pipeline.units import from_si, to_si

SIZE_QUANTITIES = {"length", "width", "height", "diameter", "radius", "thickness", "size"}

# Tunable measurement choices (swept in experiments/solver_sweep.py).
SETTINGS = {
    "width": "bbox",          # "bbox" = horizontal extent, "minor" = short principal axis
    "smooth_s": 0.25,         # Savitzky-Golay window (s) for path length / mean speed
    "mean_speed": "auto",     # "path" = smoothed path length / time, "chord" = displacement / time,
                              # "auto" = chord when the path is nearly straight
    "straight_ratio": 0.8,    # auto: chord / path length above this counts as straight
    "accel_half_s": 0.35,     # half-window (s) for instantaneous acceleration fits
    "min_ref_motion_px": 20,  # a speed/acceleration prior needs the reference to actually move
    "max_accel_rel_se": 0.25,  # acceleration prior: reference acceleration must be known to +-25%
}


class SolveError(Exception):
    pass


class Track:
    """Trajectory of one object: times t (N,) and positions pts (N, D) - pixels in 2D, metres in 3D."""

    def __init__(self, t: np.ndarray, pts: np.ndarray, fps: float):
        if len(t) < 1:
            raise SolveError("object never segmented")
        self.t, self.pts, self.fps = np.asarray(t, float), np.asarray(pts, float), fps

    @classmethod
    def from_feats(cls, feats: np.ndarray, fps: float) -> "Track":
        """Pixel-centroid track from per-frame mask features (frames where the mask is present)."""
        present = feats[:, F["present"]] == 1
        return cls(np.flatnonzero(present) / fps, feats[present][:, [F["cx"], F["cy"]]], fps)

    def window(self, a: float, b: float, min_pts: int = 3):
        sel = (self.t >= a - 1e-6) & (self.t <= b + 1e-6)
        if sel.sum() < min_pts:
            raise SolveError(f"only {sel.sum()} tracked frames in [{a:.2f}, {b:.2f}]s")
        return self.t[sel], self.pts[sel]

    def local_fit(self, t0: float, half: float, deg: int = 2) -> np.ndarray:
        """Per-coordinate polynomial fit around t0; widens the window until enough points."""
        for _ in range(6):
            sel = np.abs(self.t - t0) <= half
            if sel.sum() >= deg + 3:
                break
            half *= 1.5
        else:
            raise SolveError(f"not enough tracked frames near t={t0:.2f}s")
        tt = self.t[sel] - t0
        return np.stack([np.polyfit(tt, self.pts[sel, i], deg) for i in range(self.pts.shape[1])])  # (D, deg+1)

    def velocity_at(self, t0: float) -> float:
        coef = self.local_fit(t0, max(0.2, 3 / self.fps))
        return float(np.linalg.norm(coef[:, -2]))

    def accel_at(self, t0: float) -> float:
        coef = self.local_fit(t0, max(SETTINGS["accel_half_s"], 5 / self.fps))
        return float(np.linalg.norm(2 * coef[:, 0]))

    def fit_quality(self, a: float, b: float) -> tuple[float, float]:
        """(R^2 of a quadratic, R^2 of a linear) fit of position vs time over [a, b], all coordinates pooled."""
        t, p = self.window(a, b, min_pts=5)
        tt = t - t.mean()
        ss_tot = float(((p - p.mean(0)) ** 2).sum()) + 1e-12
        r2 = []
        for deg in (2, 1):
            res = sum(float(((np.polyval(np.polyfit(tt, p[:, i], deg), tt) - p[:, i]) ** 2).sum())
                      for i in range(p.shape[1]))
            r2.append(1 - res / ss_tot)
        return r2[0], r2[1]

    def accel_rel_se(self, a: float, b: float) -> float:
        """Relative standard error of the acceleration magnitude from a quadratic fit over [a, b]."""
        t, p = self.window(a, b, min_pts=5)  # same minimum as accel_window
        tt = t - t.mean()
        acc, var = [], []
        for i in range(p.shape[1]):
            coef, cov = np.polyfit(tt, p[:, i], 2, cov=True)
            acc.append(2 * coef[0])
            var.append(4 * cov[0, 0])
        acc, var = np.array(acc), np.array(var)
        mag = float(np.linalg.norm(acc))
        return float(np.sqrt((acc ** 2 * var).sum()) / mag ** 2) if mag > 0 else np.inf

    def check_accelerating(self, a: float, b: float):
        """An acceleration prior sets the scale only if the reference's acceleration is measured
        precisely (the scale error equals the acceleration error). A held or bouncing 'falling
        object', or a track with no visible curvature, gives a 10-100x wrong scale otherwise."""
        rel = self.accel_rel_se(a, b)
        if rel > SETTINGS["max_accel_rel_se"]:
            raise SolveError(f"reference acceleration too uncertain (relative SE {rel:.2f})")

    def accel_window(self, a: float, b: float) -> float:
        t, p = self.window(a, b, min_pts=5)
        coef = np.stack([np.polyfit(t - t.mean(), p[:, i], 2) for i in range(p.shape[1])])
        return float(np.linalg.norm(2 * coef[:, 0]))

    def path_length(self, a: float, b: float) -> tuple[float, float]:
        """Smoothed path length and elapsed time within [a, b]."""
        t, p = self.window(a, b)
        # Window in *samples*: long clips are tracked on a strided subset of frames.
        spacing = np.median(np.diff(t)) if len(t) > 1 else 1 / self.fps
        win = int(round(SETTINGS["smooth_s"] / spacing)) | 1
        if len(t) > win >= 5:
            p = savgol_filter(p, win, 2, axis=0)
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))), float(t[-1] - t[0])

    def mean_speed(self, a: float, b: float) -> float:
        """Average speed over [a, b]. Mask jitter inflates path length, so for (nearly) straight
        motion displacement / time is more accurate; curved paths (orbits) need path length."""
        t, p = self.window(a, b)
        chord = float(np.linalg.norm(p[-1] - p[0]))
        length, dt = self.path_length(a, b)
        mode = SETTINGS["mean_speed"]
        if mode == "chord" or (mode == "auto" and chord >= SETTINGS["straight_ratio"] * length):
            length = chord
        if dt <= 0:
            raise SolveError("zero-length time window")
        return length / dt

    def extent(self) -> float:
        """Diagonal of the bounding box of the whole trajectory (how far the object moves)."""
        return float(np.linalg.norm(self.pts.max(0) - self.pts.min(0)))

    def path_diameter(self, a: float, b: float) -> float:
        """Diameter of a circle fitted to the (2D) trajectory (works for partial orbits)."""
        _, p = self.window(a, b, min_pts=5)
        x, y = p[:, 0], p[:, 1]
        A = np.stack([x, y, np.ones_like(x)], 1)
        cx2, cy2, c = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)[0]
        r2 = c + (cx2 / 2) ** 2 + (cy2 / 2) ** 2
        if r2 <= 0:
            raise SolveError("circle fit failed")
        return 2 * float(np.sqrt(r2))

    def position_at(self, t0: float) -> np.ndarray:
        i = int(np.argmin(np.abs(self.t - t0)))
        if abs(self.t[i] - t0) > 2.5 / self.fps:
            raise SolveError(f"object not tracked near t={t0:.2f}s")
        return self.pts[i]


def size_series(feats: np.ndarray, quantity: str) -> tuple[np.ndarray, np.ndarray]:
    """(frame indices, per-frame pixel size) over frames where the object is fully in view
    (all present frames if it always touches the border)."""
    present = feats[:, F["present"]] == 1
    inner = present & (feats[:, F["touches_border"]] == 0)
    use = inner if inner.any() else present
    if not use.any():
        raise SolveError("object never segmented")
    f = feats[use]
    vals = {
        "length": f[:, F["major"]], "size": f[:, F["major"]], "thickness": f[:, F["minor"]],
        "width": f[:, F["minor"]] if SETTINGS["width"] == "minor" else f[:, F["x2"]] - f[:, F["x1"]],
        "height": f[:, F["y2"]] - f[:, F["y1"]],
        "diameter": f[:, F["eq_diam"]], "radius": f[:, F["eq_diam"]] / 2,
    }[quantity]
    return np.flatnonzero(use), vals


def pixel_size(feats: np.ndarray, quantity: str) -> float:
    return float(np.median(size_series(feats, quantity)[1]))


def _resolve(tw, duration):
    """[start, end] with -1 meaning end of clip -> concrete seconds (or None)."""
    if tw is None:
        return None
    a, b = (duration if v is not None and v < 0 else v for v in tw)
    return [max(0.0, min(a, duration)), max(0.0, min(b, duration))]


def _is_instant(tw):
    return tw is not None and abs(tw[1] - tw[0]) < 1e-6


def accel_check_window(tw, a: float, b: float) -> tuple[float, float]:
    """Window over which to verify that the reference accelerates: +-0.5 s around an instant."""
    return (a - 0.5, a + 0.5) if _is_instant(tw) else (a, b)


def meters_per_pixel(rec: dict, ref: np.ndarray, fps: float, duration: float) -> tuple[float, float]:
    prior_si = abs(to_si(rec["prior_value"], rec["prior_unit"]))
    q, tw = rec["ref_quantity"], _resolve(rec["ref_time"], duration)
    if q in SIZE_QUANTITIES:
        px = pixel_size(ref, q)
    else:
        tr = Track.from_feats(ref, fps)
        if tr.extent() < SETTINGS["min_ref_motion_px"]:
            raise SolveError("reference barely moves; motion prior unusable")
        a, b = tw if tw else (0.0, duration)
        if q == "speed":
            px = tr.velocity_at(a) if _is_instant(tw) else tr.mean_speed(a, b)
        else:
            tr.check_accelerating(*accel_check_window(tw, a, b))
            px = tr.accel_at(a) if _is_instant(tw) else tr.accel_window(a, b)
    if not np.isfinite(px) or px <= 1e-6:
        raise SolveError(f"reference measured {px:.3g} px")
    return prior_si / px, px


def target_pixels(rec: dict, feats: list[np.ndarray], fps: float, duration: float) -> float:
    q, tw = rec["target_quantity"], _resolve(rec["target_time"], duration)
    if q in SIZE_QUANTITIES:
        return pixel_size(feats[0], q)
    if q == "distance":
        if len(feats) != 2:
            raise SolveError("distance needs two objects")
        ta, tb = Track.from_feats(feats[0], fps), Track.from_feats(feats[1], fps)
        if tw is not None:
            return float(np.linalg.norm(ta.position_at(tw[0]) - tb.position_at(tw[0])))
        common = (feats[0][:, F["present"]] == 1) & (feats[1][:, F["present"]] == 1)
        if not common.any():
            raise SolveError("objects never visible together")
        d = feats[0][common][:, [F["cx"], F["cy"]]] - feats[1][common][:, [F["cx"], F["cy"]]]
        return float(np.median(np.hypot(*d.T)))
    tr = Track.from_feats(feats[0], fps)
    a, b = tw if tw else (float(tr.t[0]), float(tr.t[-1]))
    if q == "speed":
        return tr.velocity_at(a) if _is_instant(tw) else tr.mean_speed(a, b)
    if q == "acceleration":
        return tr.accel_at(a) if _is_instant(tw) else tr.accel_window(a, b)
    if q == "displacement":
        return float(np.linalg.norm(tr.position_at(b) - tr.position_at(a)))
    if q == "path_length":
        return tr.path_length(a, b)[0]
    if q == "path_diameter":
        return tr.path_diameter(a, b)
    raise SolveError(f"unsupported target quantity {q!r}")


def solve(rec: dict, ref: np.ndarray, targets: list[np.ndarray], fps: float, n_frames: int) -> dict:
    duration = (n_frames - 1) / fps
    mpp, ref_px = meters_per_pixel(rec, ref, fps, duration)
    tgt_px = target_pixels(rec, targets, fps, duration)
    value = from_si(tgt_px * mpp, rec["target_unit"])
    return {"value": float(value), "m_per_px": mpp, "ref_px": ref_px, "target_px": tgt_px}
