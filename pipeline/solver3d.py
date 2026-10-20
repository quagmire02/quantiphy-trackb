"""3D solver: pinhole camera + the given object-to-camera distances.

Model: principal point at the image centre, square pixels, unknown focal length f (pixels of
the tracking frames). A pixel (u, v) at depth Z maps to P = Z * [(u-cx)/f, (v-cy)/f, 1].
The dataset gives Euclidean object-camera distances d (not depth along the optical axis), so
Z = d / sqrt(1 + ((u-cx)^2 + (v-cy)^2) / f^2).

f is calibrated per clip so the reference object reproduces the prior (its size, speed or
acceleration); every target quantity is then computed in metres from its 3D track.
"""
import difflib
import re

import numpy as np
from scipy.optimize import brentq

from pipeline.geometry import F
from pipeline.solver import (SIZE_QUANTITIES, SolveError, Track, _is_instant, _resolve,  # noqa: F401
                             accel_check_window, size_series, target_accel)
from pipeline.units import from_si, to_si

SETTINGS = {
    "depth_interp": "linear",  # >=2 timed distances: "linear" in time, or "size" (apparent-size scaling)
    "f_range": (0.1, 30.0),    # search range for f, in multiples of the frame width
    "depth_outlier_factor": 20.0,  # distances this far from the clip's median distance are typos
    "moge_transfer": True,     # objects without a stated distance get one via MoGe depth ratios
    "max_anchor_spread": 2.0,  # transfer: known objects / frames must agree within this factor
    "fuzzy_name_match": 0.85,  # letters-only similarity to match 'whiteboard' with 'white_board'
    "f_source": "prior",       # "prior" = calibrate f from the prior; "moge" = MoGe-2's focal length only;
                               # "blend" = geometric mean of both (motion priors only, else "prior")
    "f_fallback_moge": True,   # prior calibration impossible -> use MoGe-2's focal length
    "min_f_over_w": 0.3,       # calibrated f below this (field of view > 118 deg) is implausible -> MoGe-2 f
    "min_f_sensitivity": 0.0,  # |d log(prior) / d log f| below this: the prior barely fixes f -> MoGe-2 f
    "equal_depth_fallback": True,  # no stated or transferred distance -> the clip's median stated distance
    # Known camera (all "lab" clips come from one rig): f is the most likely value given BOTH the camera
    # (tight: sd of log f) and the prior's measurement (looser: motion blur inflates a moving reference
    # ball by ~6%, motion toward the camera barely constrains f). FOV estimated on validation lab clips.
    "loom": "raw",             # (test 2026-10-10: gated made val S3 -1.7 and variant agreement worse)
                               # < 2 timed distances: "gated" = apparent-size depth only for a clear, smooth
                               # size change (else constant); "raw" = old behaviour (1 timed: always raw size)
    "loom_min_change": 0.08,   # smoothed change of log size needed (~8%)
    "loom_snr": 4.0,           # ... and this many times the residual sd (wing flaps, gait)
    "lab_fov_deg": 84.0,       # horizontal field of view of the lab camera (None = not used); validation:
                               # GT-implied 86 deg (clips 84-90), lab-item score flat over 82-86 deg
    "cam_sig_log_f": 0.05,
    "prior_sig_log": 0.20,
    "max_f_disagreement": 0.0,  # motion prior: calibrated f vs (bias-corrected) MoGe-2 f apart by more
                                # than this factor -> the calibration broke, use MoGe-2 f (0 = off)
}

_NUM = r"\d*\.?\d+"
DEPTH_LINE = re.compile(rf"(?:t\s*=\s*({_NUM})\s*s?\s*,\s*)?distance_(.+?)(?:_?camera)?\s*=\s*({_NUM})\s*(cm|mm|m)?",
                        re.I)
_SYNONYMS = {"human": "person", "man": "person", "woman": "person", "people": "person", "pedestrian": "person",
             "pingpong": "ping pong", "soccerball": "soccer ball", "notebook": "note book", "centeral": "central"}
# "a"/"b" are kept: they often identify objects ("ball_a").
_STOP = {"the", "of", "and", "camera", "distance", "to", "in", "on", "with"}


def _tokens(text: str) -> set[str]:
    t = re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("_", " "))
    words = " ".join(_SYNONYMS.get(w, w) for w in t.split()).split()
    return {w.rstrip("s") if len(w) > 3 else w for w in words if w not in _STOP}


def parse_depth(text) -> list[dict]:
    """'t=0s, distance_boat_camera = 11.8 m' lines -> [{'name', 'tokens', 't' (s or None), 'd' (m)}]."""
    out = []
    for line in str(text or "").splitlines():
        m = DEPTH_LINE.search(line)
        if not m:
            continue
        name = re.sub(r"^camera_|_camera$", "", m.group(2).strip())
        d = float(m.group(3)) * {"cm": 0.01, "mm": 0.001}.get((m.group(4) or "m").lower(), 1.0)
        out.append({"name": name, "tokens": _tokens(name), "t": None if m.group(1) is None else float(m.group(1)),
                    "d": d})
    # Drop annotation typos such as "distance_white_cup_camera = 8420m" in a clip whose other
    # distances are ~1 m (objects in one short clip lie within a factor ~20 of each other).
    if len(out) >= 2:
        med = float(np.median([e["d"] for e in out]))
        out = [e for e in out if med / SETTINGS["depth_outlier_factor"] <= e["d"] <= med * SETTINGS["depth_outlier_factor"]]
    return out


def _compacts(text: str) -> set[str]:
    """Letters-only spellings, in the given and in sorted word order ('white_board' -> whiteboard, boardwhite)."""
    words = [re.sub(r"[^a-z]", "", w) for w in re.split(r"[\s_]+", text.lower().split("#")[0])]
    words = [w for w in words if w and w not in _STOP]
    return {"".join(words), "".join(sorted(words))}


def match_depth(entries: list[dict], phrase: str) -> list[dict]:
    """Depth entries whose name best matches an object phrase (all timestamps of that name).
    Jaccard overlap, so 'ball b' prefers 'ball_b' over 'ball_a' and 'left tennis ball' prefers
    'left_tennis_ball' over 'right_tennis_ball'. If nothing overlaps, fall back to comparing
    letters-only spellings, so compound words and typos still match: 'whiteboard' ~ 'white_board',
    'shoppingbag' ~ 'shopping_bag', 'soccor ball' ~ 'soccer_ball'."""
    want = _tokens(phrase.split("#")[0])
    best, best_score = None, 0.0
    for name in dict.fromkeys(e["name"] for e in entries):
        toks = next(e["tokens"] for e in entries if e["name"] == name)
        if not toks or not want:
            continue
        score = len(want & toks) / len(want | toks)
        if score > best_score:
            best, best_score = name, score
    if best_score >= 0.5:
        return [e for e in entries if e["name"] == best]
    best, best_score = None, 0.0
    for name in dict.fromkeys(e["name"] for e in entries):
        score = max(difflib.SequenceMatcher(None, a, b).ratio()
                    for a in _compacts(phrase) for b in _compacts(name))
        if score > best_score:
            best, best_score = name, score
    if best_score >= SETTINGS["fuzzy_name_match"]:
        return [e for e in entries if e["name"] == best]
    # Compound nouns: 'basketball' / 'football' is the stated 'ball' when exactly one stated name is
    # the tail of one of the phrase's words (keys hold sorted words, so any word, not just the last).
    words = {re.sub(r"[^a-z]", "", w) for w in phrase.lower().split("#")[0].split()}
    tails = [name for name in dict.fromkeys(e["name"] for e in entries)
             if len(c := re.sub(r"[^a-z]", "", name.lower())) >= 3
             and any(w != c and w.endswith(c) for w in words)]
    return [e for e in entries if e["name"] == tails[0]] if len(tails) == 1 else []


class Camera:
    def __init__(self, width: int, height: int, f: float):
        self.cx, self.cy, self.f = width / 2, height / 2, f

    def z_from_distance(self, d, u, v):
        return d / np.sqrt(1 + ((u - self.cx) ** 2 + (v - self.cy) ** 2) / self.f ** 2)

    def point(self, z, u, v):
        return np.stack([z * (u - self.cx) / self.f, z * (v - self.cy) / self.f, z], axis=-1)


class Object3D:
    """Tracked object + its distance anchors; yields metric 3D tracks for a given camera."""

    def __init__(self, feats: np.ndarray, anchors: list[dict], fps: float, motion: bool = False):
        """motion: the object's speed/acceleration/position is asked (not only its size)."""
        self.feats, self.anchors, self.fps, self.motion = feats, anchors, fps, motion
        self.present = np.flatnonzero(feats[:, F["present"]] == 1)
        if len(self.present) == 0:
            raise SolveError("object never segmented")
        if not anchors:
            raise SolveError("no distance given for object")
        self.u = feats[self.present, F["cx"]]
        self.v = feats[self.present, F["cy"]]
        self.t = self.present / fps
        self.scale = np.sqrt(feats[self.present, F["area"]])  # apparent size

    def _nearest(self, t0: float) -> int:
        i = int(np.argmin(np.abs(self.t - t0)))
        if abs(self.t[i] - t0) > 2.5 / self.fps:
            raise SolveError(f"object not tracked near its distance timestamp t={t0:.2f}s")
        return i

    def _smoothed_log_size(self):
        """(times, smoothed log apparent size at every tracked frame) when the object's size changes
        clearly and smoothly over the clip (motion toward / away from the camera), else None.
        Uses frames where the object is fully inside the image; a local quadratic fit (+-0.5 s) per
        frame smooths mask jitter; the change must exceed max(LOOM_MIN_CHANGE, LOOM_SNR x residual sd)."""
        inner = self.feats[self.present, F["touches_border"]] == 0
        t, ls = self.t[inner], np.log(np.maximum(self.scale[inner], 1e-6))
        if len(np.unique(t)) < 3 or np.ptp(t) < 0.2:
            return None
        sm = np.empty(len(t))
        for i, t0 in enumerate(t):
            sel = np.abs(t - t0) <= 0.5
            if sel.sum() < 3:
                sel = np.argsort(np.abs(t - t0))[:3]
            x = t[sel] - t0
            sm[i] = np.polyval(np.polyfit(x, ls[sel], min(2, len(np.unique(x)) - 1)), 0.0)
        resid = ls - sm
        sd = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
        if np.ptp(sm) < max(SETTINGS["loom_min_change"], SETTINGS["loom_snr"] * sd):
            return None
        return t, sm

    def distances(self) -> np.ndarray:
        """Euclidean camera distance at every tracked frame."""
        timed = sorted((a["t"], a["d"]) for a in self.anchors if a["t"] is not None)
        if SETTINGS["loom"] == "gated" and len(timed) < 2:
            # < 2 timed distances: follow the apparent size only when it changes clearly and smoothly
            # (range ~ 1 / size), anchored at the timed distance (or the mean untimed one); else constant.
            loom = self._smoothed_log_size()
            if loom is None:
                d0 = timed[0][1] if timed else float(np.median([a["d"] for a in self.anchors]))
                return np.full(len(self.t), float(d0))
            ts, sm = loom
            size = np.exp(np.interp(self.t, ts, sm))
            if timed:
                t0, d0 = timed[0]
                est = d0 * float(np.exp(np.interp(t0, ts, sm))) / size
            else:
                est = float(np.mean([a["d"] for a in self.anchors])) * float(np.exp(np.median(sm))) / size
            if self.motion and any(a.get("derived") for a in self.anchors) and est.max() / est.min() > 2.0:
                return np.full(len(self.t), float(timed[0][1] if timed else np.median(est)))
            return est
        if not timed:  # static distance
            return np.full(len(self.t), float(np.median([a["d"] for a in self.anchors])))
        if len(timed) >= 2 and SETTINGS["depth_interp"] == "linear":
            ts, ds = np.array(timed).T
            slope = np.polyfit(ts, ds, 1)[0] if len(ts) > 1 else 0.0
            d = np.interp(self.t, ts, ds)
            d = np.where(self.t < ts[0], ds[0] + slope * (self.t - ts[0]), d)
            d = np.where(self.t > ts[-1], ds[-1] + slope * (self.t - ts[-1]), d)
            return np.maximum(d, 1e-3)
        # Apparent-size scaling (distance inversely proportional to size), averaged over anchors.
        est = np.mean([d0 * self.scale[self._nearest(t0)] / self.scale for t0, d0 in timed], axis=0)
        # A MoGe-transferred distance scaled by apparent size explodes when the object is partly
        # masked or entering the frame (a bicycle went 44 m -> 123 m, speeds 17-88 m/s): for motion
        # quantities keep the transferred distance fixed when the scaled distances span > 2x.
        if self.motion and any(a.get("derived") for a in self.anchors) and est.max() / est.min() > 2.0:
            return np.full(len(self.t), float(timed[0][1]))
        return est

    def distance_at(self, fi: int) -> float:
        """Euclidean camera distance at frame index fi (nan if the object is not tracked there)."""
        pos = np.flatnonzero(self.present == fi)
        return float(self.distances()[pos[0]]) if len(pos) else float("nan")

    def track(self, cam: Camera) -> Track:
        z = cam.z_from_distance(self.distances(), self.u, self.v)
        return Track(self.t, cam.point(z, self.u, self.v), self.fps)

    def size(self, cam: Camera, quantity: str) -> float:
        frames, px = size_series(self.feats, quantity)
        pos = {f: i for i, f in enumerate(self.present)}
        keep = [i for i, f in enumerate(frames) if f in pos]
        if not keep:
            raise SolveError("no frame with both size and distance")
        idx = np.array([pos[frames[i]] for i in keep])
        d = self.distances()[idx]
        z = cam.z_from_distance(d, self.u[idx], self.v[idx])
        # Anchor frames are the most trustworthy; use them when the object is fully visible there.
        timed = [self._nearest(a["t"]) for a in self.anchors if a["t"] is not None]
        at_anchor = np.isin(idx, timed)
        sel = at_anchor if at_anchor.any() else np.ones(len(idx), bool)
        return float(np.median(px[keep][sel] * z[sel] / cam.f))


def transfer_anchors(key: str, anchored: dict[str, Object3D], moge: dict[int, dict[str, float]],
                     fps: float) -> list[dict]:
    """One distance anchor for an object without a stated distance. Per frame with MoGe distances:
    d_key = d_known(frame) * moge_key / moge_known for every object with a stated distance.
    Frames whose estimates disagree by more than max_anchor_spread are dropped, and so is the whole
    transfer if the per-frame values disagree that much across frames. The result is ONE anchor at
    the frame whose estimate is the median, so depth changes come from apparent size, not from MoGe
    frame-to-frame noise (joining noisy per-frame estimates invented motion)."""
    per_frame = []
    for fi, dists in sorted(moge.items()):
        if key not in dists:
            continue
        est = []
        for k, obj in anchored.items():
            if k == key or k not in dists:
                continue
            try:
                d_known = obj.distance_at(fi)
            except SolveError:  # one unusable known object must not sink the transfer
                continue
            if np.isfinite(d_known) and d_known > 0:
                est.append(d_known * dists[key] / dists[k])
        if est and max(est) / min(est) <= SETTINGS["max_anchor_spread"]:
            per_frame.append((fi, float(np.median(est))))
    if not per_frame:
        return []
    ds = np.array([d for _, d in per_frame])
    if ds.max() / ds.min() > SETTINGS["max_anchor_spread"]:
        return []
    fi, d = per_frame[int(np.argmin(np.abs(ds - np.median(ds))))]
    return [{"name": f"moge:{key}", "tokens": set(), "t": fi / fps, "d": d, "derived": True}]


def quantity_3d(q: str, tw, objs: list[Object3D], cam: Camera, duration: float, target: bool = True) -> float:
    """Requested quantity in SI units from metric 3D tracks. target=False: the reference during
    focal-length calibration (keeps the local acceleration fit, which validated better there)."""
    if q in SIZE_QUANTITIES:
        return objs[0].size(cam, q)
    if q == "distance":
        if len(objs) != 2:
            raise SolveError("distance needs two objects")
        ta, tb = objs[0].track(cam), objs[1].track(cam)
        if tw is not None:
            return float(np.linalg.norm(ta.position_at(tw[0]) - tb.position_at(tw[0])))
        common = np.intersect1d(np.round(ta.t, 6), np.round(tb.t, 6))
        if not len(common):
            raise SolveError("objects never visible together")
        return float(np.median([np.linalg.norm(ta.position_at(t) - tb.position_at(t)) for t in common]))
    tr = objs[0].track(cam)
    a, b = tw if tw else (float(tr.t[0]), float(tr.t[-1]))
    if q == "speed":
        return tr.velocity_at(a) if _is_instant(tw) else tr.mean_speed(a, b)
    if q == "acceleration":
        if not target:
            return tr.accel_at(a) if _is_instant(tw) else tr.accel_window(a, b)
        return target_accel(tr, Track.from_feats(objs[0].feats, objs[0].fps), tw, a, b)
    if q == "displacement":
        return float(np.linalg.norm(tr.position_at(b) - tr.position_at(a)))
    if q == "path_length":
        return tr.path_length(a, b)[0]
    raise SolveError(f"unsupported 3D target quantity {q!r}")


def calibrate_f(rec: dict, ref: Object3D, width: int, height: int, duration: float) -> float:
    """Focal length (px) that makes the reference reproduce the prior value."""
    prior_si = abs(to_si(rec["prior_value"], rec["prior_unit"]))
    tw = _resolve(rec["ref_time"], duration)

    def predicted(f):
        return quantity_3d(rec["ref_quantity"], tw, [ref], Camera(width, height, f), duration, target=False)

    lo, hi = SETTINGS["f_range"]
    grid = np.geomspace(lo * width, hi * width, 60)
    vals = []
    for f in grid:
        try:
            vals.append(np.log(predicted(f) / prior_si))
        except (SolveError, FloatingPointError, ValueError):
            vals.append(np.nan)
    vals = np.array(vals)
    ok = np.isfinite(vals)
    sign_change = np.flatnonzero(ok[:-1] & ok[1:] & (np.sign(vals[:-1]) != np.sign(vals[1:])))
    if not len(sign_change):
        raise SolveError("no focal length reproduces the prior")
    i = sign_change[0]
    f = float(brentq(lambda f: np.log(predicted(f) / prior_si), grid[i], grid[i + 1], xtol=1e-3))
    # How strongly the prior pins f down: d log(predicted) / d log(f) at the solution. A size prior
    # gives -1; a speed mostly toward the camera gives ~0 (the radial part comes from the stated
    # distances, not from f), and then a small measurement error becomes a large error in f.
    h = 1.1
    try:
        sens = (np.log(predicted(f * h)) - np.log(predicted(f / h))) / (2 * np.log(h))
    except (SolveError, FloatingPointError, ValueError):
        sens = float("nan")
    if abs(sens) < SETTINGS["min_f_sensitivity"]:
        raise SolveError(f"focal length poorly determined by the prior (sensitivity {sens:.2f})")
    return f


def map_f(rec: dict, ref: Object3D, width: int, height: int, duration: float, f_cam: float) -> float:
    """Focal length maximising N(log f; log f_cam, cam_sig) x N(log(prior measured / prior); 0, prior_sig):
    a known camera, unless the prior pins f down much more tightly. f_cam if the prior never evaluates."""
    prior_si = abs(to_si(rec["prior_value"], rec["prior_unit"]))
    tw = _resolve(rec["ref_time"], duration)
    lf = np.linspace(np.log(f_cam / 3), np.log(f_cam * 3), 161)
    cost = np.full(len(lf), np.inf)
    for i, x in enumerate(lf):
        try:
            q = quantity_3d(rec["ref_quantity"], tw, [ref], Camera(width, height, float(np.exp(x))), duration,
                            target=False)
        except (SolveError, FloatingPointError, ValueError):
            continue
        if q > 0 and np.isfinite(q):
            cost[i] = ((x - np.log(f_cam)) / SETTINGS["cam_sig_log_f"]) ** 2 \
                + (np.log(q / prior_si) / SETTINGS["prior_sig_log"]) ** 2
    return float(np.exp(lf[int(np.argmin(cost))])) if np.isfinite(cost).any() else float(f_cam)


def solve_3d(rec: dict, feats: dict[str, np.ndarray], keys: tuple[str, list[str]], depth_text, fps: float,
             n_frames: int, width: int, height: int, moge: dict[int, dict[str, float]] | None = None,
             moge_f: float | None = None, known_fov_deg: float | None = None) -> dict:
    """feats: every tracked object of the clip (key -> per-frame features); moge: frame -> {key: MoGe
    distance} (optional) used to give objects without a stated distance one via transfer_anchors();
    moge_f: MoGe-2's focal length (px of the tracking frames), an alternative to calibrating f;
    known_fov_deg: the camera's horizontal field of view when known (lab rig): f = map_f(...)."""
    duration = (n_frames - 1) / fps
    entries = parse_depth(depth_text)
    if not entries:
        raise SolveError("no parsable distances")
    ref_key, tgt_keys = keys
    anchored = {}
    for k, f in feats.items():
        if match_depth(entries, k):
            try:
                anchored[k] = Object3D(f, match_depth(entries, k), fps)
            except SolveError:
                pass

    def make(k: str, quantity: str) -> Object3D:
        anchors = match_depth(entries, k)
        if not anchors and moge and SETTINGS["moge_transfer"]:
            anchors = transfer_anchors(k, anchored, moge, fps)
        if not anchors and SETTINGS["equal_depth_fallback"]:
            # Last resort: assume the object is as far as the clip's stated objects typically are
            # (objects in one short clip are mostly at similar depth; within 25% for ~73% of pairs on
            # this data). Static, so depth noise can't invent motion.
            anchors = [{"name": "clip-median", "tokens": set(), "t": None, "derived": True,
                        "d": float(np.median([e["d"] for e in entries]))}]
        return Object3D(feats[k], anchors, fps, motion=quantity not in SIZE_QUANTITIES)

    def from_prior() -> tuple[float, Object3D]:
        if rec["ref_quantity"] == "acceleration":  # same sanity check as 2D, on the pixel track
            tw = _resolve(rec["ref_time"], duration)
            a, b = tw if tw else (0.0, duration)
            Track.from_feats(feats[ref_key], fps).check_accelerating(*accel_check_window(tw, a, b))
        ref = make(ref_key, rec["ref_quantity"])
        return calibrate_f(rec, ref, width, height, duration), ref

    usable_moge = moge_f is not None and np.isfinite(moge_f) and moge_f > 0
    ref, f_source = None, "prior"
    if known_fov_deg:
        f_cam = (width / 2) / np.tan(np.radians(known_fov_deg) / 2)
        try:
            if rec["ref_quantity"] == "acceleration":
                tw = _resolve(rec["ref_time"], duration)
                a, b = tw if tw else (0.0, duration)
                Track.from_feats(feats[ref_key], fps).check_accelerating(*accel_check_window(tw, a, b))
            ref = make(ref_key, rec["ref_quantity"])
            f, f_source = map_f(rec, ref, width, height, duration, f_cam), "camera+prior"
        except SolveError:  # the prior can't be measured: the camera alone sets the scale
            f, f_source = f_cam, "camera"
    elif SETTINGS["f_source"] == "moge" and usable_moge:
        f, f_source = moge_f, "moge"
    else:
        try:
            f, ref = from_prior()
            if (SETTINGS["f_fallback_moge"] and usable_moge and f / width < SETTINGS["min_f_over_w"]
                    and moge_f / width >= SETTINGS["min_f_over_w"]):
                f, f_source = moge_f, "moge"  # no real camera here is that wide: the calibration is broken
            elif (SETTINGS["f_fallback_moge"] and usable_moge and SETTINGS["max_f_disagreement"] > 0
                  and rec["ref_quantity"] not in SIZE_QUANTITIES
                  and max(f / moge_f, moge_f / f) > SETTINGS["max_f_disagreement"]):
                f, f_source = moge_f, "moge"  # a noisy speed/acceleration calibration far from MoGe-2's
            elif SETTINGS["f_source"] == "blend" and usable_moge and rec["ref_quantity"] not in SIZE_QUANTITIES:
                f, f_source = float(np.sqrt(f * moge_f)), "blend"
        except SolveError:
            if not (SETTINGS["f_fallback_moge"] and usable_moge):
                raise
            f, f_source = moge_f, "moge"
    cam = Camera(width, height, f)
    objs = [make(k, rec["target_quantity"]) for k in tgt_keys]
    value_si = quantity_3d(rec["target_quantity"], _resolve(rec["target_time"], duration), objs, cam, duration)
    derived = any(a.get("derived") for o in ([ref] if ref else []) + objs for a in o.anchors)
    return {"value": float(from_si(value_si, rec["target_unit"])), "f_px": f, "f_over_w": f / width,
            "f_source": f_source, "ref_depth": ref.anchors[0]["name"] if ref else "",
            "tgt_depth": ",".join(o.anchors[0]["name"] for o in objs), "moge_used": derived}
