"""Post-solve plausibility guards: reject measurements whose implied scale is physically absurd.
Rejected items fall back to the VLM answer.

G1  person height outside 0.8-2.4 m -> the clip's calibration is broken (one metres-per-pixel scale
    in 2D, one focal length in 3D, both set from the prior) -> reject every measured item of the clip.
G2  same-kind ratio: a size vs a size prior, speed vs speed, acceleration vs acceleration more than
    1000x apart is implausible within one short clip -> reject the item.
G3  gravity prior (says "gravity", or names no object, with |a| ~ 9.8 m/s^2): the falling reference
    itself cannot be faster than g * (t + 2 s) at time t (allowing a throw of up to ~20 m/s). A
    violation means the calibration from it is wrong -> reject every measured item of the clip
    (2D and 3D). Other objects in the clip (e.g. a plane) are not bound by free fall.
"""
import re

import numpy as np
import pandas as pd

from pipeline.units import to_si

PERSON = re.compile(r"\b(person|people|man|woman|boy|girl|human|pedestrian|player|child|kid|lady|guy|passer|"
                    r"receiver)\b", re.I)
LENGTH_Q = {"length", "width", "height", "diameter", "radius", "thickness", "size", "distance", "displacement",
            "path_length", "path_diameter"}
PERSON_HEIGHT_M = (0.8, 2.4)
MAX_SAME_KIND_RATIO = 1e3
GRAVITY = (9.0, 10.5)       # |prior acceleration| treated as free fall
THROW_ALLOWANCE_S = 2.0     # initial speed allowance, in seconds of free fall (~20 m/s)
DEFAULT_TIME_S = 5.0        # time used for whole-clip averages


def _kind(q):
    return "length" if q in LENGTH_Q else q


def _words(phrase: str) -> set[str]:
    return set(str(phrase).lower().split()) - {"the", "of", "a", "an"}


def _same_object(a: str, b: str) -> bool:
    """'ball' and 'falling ball' may be the same object: word sets nested either way."""
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and (wa <= wb or wb <= wa)


def _is_gravity_prior(rec: dict, prior_si: float, prior_text: str) -> bool:
    """'gravity acc = 9.8', 'acceleration = 9.8' (no object named) - not 'acceleration of the car = 10'."""
    return (rec["ref_quantity"] == "acceleration" and GRAVITY[0] <= prior_si <= GRAVITY[1]
            and ("grav" in str(prior_text).lower() or rec.get("ref_object_source") == "vlm"))


def apply_guards(df: pd.DataFrame, parsed: dict, sol: pd.DataFrame) -> pd.DataFrame:
    sol = sol.copy()
    info = df.set_index("id")[["video_id", "prior"]]
    si = {}
    for r in sol.itertuples():
        rec = parsed.get(r.id)
        if rec and np.isfinite(r.value):
            si[r.id] = to_si(r.value, rec["target_unit"])

    reject, bad_clips = {}, {}
    for i, v in si.items():
        rec, vid = parsed[i], info.loc[i, "video_id"]
        if (rec["target_quantity"] == "height" and any(PERSON.search(t) for t in rec["targets"])
                and not PERSON_HEIGHT_M[0] <= v <= PERSON_HEIGHT_M[1]):
            bad_clips[vid] = "guard: implausible person height in clip"
        prior_si = abs(to_si(rec["prior_value"], rec["prior_unit"]))
        if (_is_gravity_prior(rec, prior_si, info.loc[i, "prior"]) and rec["target_quantity"] == "speed"
                and len(rec["targets"]) == 1 and _same_object(rec["targets"][0], rec["ref_object"])):
            tw = rec["target_time"]
            t = tw[1] if tw and tw[1] >= 0 else DEFAULT_TIME_S
            if v > prior_si * (t + THROW_ALLOWANCE_S):
                bad_clips[vid] = f"guard: {v:.3g} m/s is faster than free fall allows"
        if _kind(rec["target_quantity"]) == _kind(rec["ref_quantity"]):
            ratio = v / prior_si
            if not 1 / MAX_SAME_KIND_RATIO <= ratio <= MAX_SAME_KIND_RATIO:
                reject[i] = f"guard: {ratio:.2g}x the prior"
    for i in si:
        vid = info.loc[i, "video_id"]
        if vid in bad_clips:
            reject.setdefault(i, bad_clips[vid])
    mask = sol["id"].isin(reject)
    sol.loc[mask, "reason"] = sol.loc[mask, "id"].map(reject)
    sol.loc[mask, "value"] = np.nan
    return sol
