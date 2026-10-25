"""Conservative recoveries using the question and supplied metadata only."""
import json
import re

import numpy as np

from pipeline.solver3d import match_depth, parse_depth
from pipeline.units import _NUM, _UNIT_TOKEN, from_si, question_unit, unit_factor


def direct_value(reply, question):
    """Parse an answer together with its own unit; abstain on ambiguous explanations."""
    text = str(reply).strip().replace('**', '').replace('\\,', '')
    text = text.replace('\\mathrm{', '').replace('\\text{', '').replace('}', '') if '\\' in text else text
    target = question_unit(question)
    # A final-answer marker limits parsing to the answer rather than numbers in the reasoning.
    marks = list(re.finditer(r'(?:final\s+answer|the\s+answer\s+is|answer)\s*[:=]?\s*', text, re.I))
    if marks:
        text = text[marks[-1].end():]
    pairs = []
    for match in re.finditer(_NUM, text):
        tail = text[match.end():].lstrip()
        unit = _UNIT_TOKEN.match(tail)
        if unit:
            value = float(match[0])
            try:
                f, p = unit_factor(unit[0])
                if target:
                    ft, pt = unit_factor(target)
                    if p != pt:
                        continue
                    value *= f / ft
            except KeyError:
                continue
            pairs.append(value)
    if len(pairs) == 1:
        result = pairs[0]
    elif len(pairs) > 1 and np.allclose(pairs, pairs[0], rtol=1e-6):
        result = pairs[0]
    elif re.fullmatch(rf'\s*({_NUM})\s*[.!]?\s*', text):
        result = float(re.fullmatch(rf'\s*({_NUM})\s*[.!]?\s*', text)[1])
    else:
        return float('nan')
    return float(result) if np.isfinite(result) and result > 0 else float('nan')


def camera_distance(row, parsed):
    """Answer explicitly supplied camera ranges; no assumed depth or extrapolation."""
    question = str(row.question)
    if str(row.video_type)[1] != '3' or not re.search(r'\bdistance\s+from\s+(?:the\s+)?(?:shooting\s+)?camera\b', question, re.I):
        return None
    targets = [x for x in (parsed or {}).get('targets', []) if x.lower().strip() not in ('camera', 'shooting camera')]
    if len(targets) != 1:
        return None
    entries = parse_depth(row.depth_info)
    anchors = match_depth(entries, targets[0])
    if not anchors:
        return None
    timed = sorted((x['t'], x['d']) for x in anchors if x['t'] is not None)
    tw = parsed.get('target_time')
    if tw is None or tw[0] < 0 or tw[0] != tw[1]:
        return None
    time = tw[0]
    if timed:
        ts, ds = np.array(timed).T
        if len(set(ts)) != len(ts) or time < ts.min() or time > ts.max():
            return None
        value = float(np.interp(time, ts, ds))
    else:
        ds = [x['d'] for x in anchors]
        if not np.allclose(ds, ds[0]):
            return None
        value = ds[0]
    return from_si(value, question_unit(question))


def prior_scale_fallback(row):
    """Explicit last-resort guess from this question's prior and video duration only."""
    from pipeline.parse import first_prior, prior_description, prior_quantity
    from pipeline.units import parse_prior_value, to_si
    value, unit = parse_prior_value(first_prior(row.prior))
    if value is None or not np.isfinite(value) or value <= 0:
        return None
    quantity = prior_quantity(prior_description(row.prior))
    power = {'speed': 1, 'acceleration': 2}.get(quantity, 0)
    target = question_unit(row.question)
    if not target:
        return None
    target_power = unit_factor(target)[1]
    scale = to_si(value, unit)
    if power != target_power:
        import cv2
        cap = cv2.VideoCapture(row.video_path)
        frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if frames <= 1 or fps <= 0:
            return None
        duration = (frames-1) / fps
        scale *= duration ** (power-target_power)
    result = from_si(scale, target)
    return float(result) if np.isfinite(result) and result > 0 else None
