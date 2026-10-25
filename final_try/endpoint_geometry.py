"""CPU geometry for a selective 2D endpoint experiment, independent of SAM target masks."""
import json
import re

import numpy as np

from pipeline.units import from_si

SIZE = {'length', 'width', 'height', 'diameter', 'radius', 'thickness', 'size', 'distance'}


def decode_points(reply, require_reference=True):
    for start, char in enumerate(str(reply)):
        if char != '{':
            continue
        try:
            obj, _ = json.JSONDecoder().raw_decode(reply[start:])
            out = {}
            for key in ['target'] + (['reference'] if require_reference else []):
                p = np.asarray(obj[key], dtype=float)
                if p.shape != (2, 2) or not np.isfinite(p).all() or (p < 0).any() or (p > 1000).any():
                    raise ValueError('Invalid points')
                if np.linalg.norm(p[1]-p[0]) <= 0:
                    raise ValueError('Coincident points')
                out[key] = p
            return out
        except (ValueError, KeyError, TypeError):
            continue
    return None


def native_points(points, box):
    x0, y0, x1, y1 = box
    return np.asarray(points, float) / 1000 * [x1-x0, y1-y0] + [x0, y0]


def crop_box(points, width, height, margin=.35):
    p = np.concatenate(list(points.values()))
    low, high = p.min(axis=0), p.max(axis=0)
    pad = np.maximum((high-low)*margin, 24)
    low = np.maximum(0, np.floor(low-pad)).astype(int)
    high = np.minimum([width, height], np.ceil(high+pad)).astype(int)
    return (int(low[0]), int(low[1]), int(high[0]), int(high[1]))


def span(points):
    return float(np.linalg.norm(np.asarray(points)[1]-np.asarray(points)[0]))


def estimate(points, prior_si, target_unit, m_per_native_px=None):
    target_px = span(points['target'])
    if target_px < 2:
        raise ValueError('Target endpoints too close')
    if m_per_native_px is None:
        ref_px = span(points['reference'])
        if ref_px < 2 or not np.isfinite(prior_si) or prior_si <= 0:
            raise ValueError('Reference endpoints/scale unusable')
        m_per_native_px = prior_si / ref_px
    if not np.isfinite(m_per_native_px) or m_per_native_px <= 0:
        raise ValueError('Scale unusable')
    value = from_si(target_px*m_per_native_px, target_unit)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Non-positive endpoint estimate')
    return float(value)


def agree(a, b, factor=1.2):
    return bool(np.isfinite(a) and np.isfinite(b) and min(a, b) > 0 and max(a, b)/min(a, b) <= factor)


def points_agree(first, second, floor_px=6., relative=.2):
    """Check spatial identity too; swapped endpoint order is equivalent."""
    if set(first) != set(second):
        return False
    for key in first:
        a, b = np.asarray(first[key]), np.asarray(second[key])
        error = min(float(np.linalg.norm(a-b, axis=1).max()), float(np.linalg.norm(a-b[::-1], axis=1).max()))
        if error > max(floor_px, relative*span(a)):
            return False
    return True


def consensus(values, factor=1.25, min_frames=2):
    values = np.asarray(values, float)
    if len(values) < min_frames or not np.isfinite(values).all() or (values <= 0).any():
        return None
    median = float(np.median(values))
    # Require ALL accepted frame readings to agree, not a chosen convenient subset.
    return median if all(agree(v, median, factor) for v in values) else None


def frame_indices(total, fps, target_time=None, count=3):
    if total < 3 or fps <= 0:
        return []
    if target_time is not None:
        if len(target_time) != 2 or target_time[0] != target_time[1]:
            return []  # Static endpoint experiment does not average a specified interval.
        t = (total-1)/fps if target_time[0] < 0 else target_time[0]
        if t < 0 or t > (total-1)/fps + 1/fps:
            return []
        center = int(np.clip(round(t*fps), 0, total-1))
        return sorted(set([max(0, center-1), center, min(total-1, center+1)]))
    return sorted(set(np.linspace(.2*(total-1), .8*(total-1), count).round().astype(int).tolist()))


def request_text(row, rec, motion_scale=False):
    return (f'ONE question: {row.question}\nGiven prior: {row.prior}\n'
            f'Reference object: {rec["ref_object"]}. Reference dimension to mark: {rec["ref_quantity"]}.\n'
            f'Targets: {json.dumps(rec.get("targets", []))}. Requested target quantity: {rec["target_quantity"]}.\n'
            'Mark the TWO physical endpoints delimiting the requested target dimension or gap. '
            'For a lane/tile/marking, identify the actual boundaries requested; do not surround the whole surface. '
            'For a gap between objects, use the requested gap boundaries, not an unrelated background edge. '
            'Respect whether the question asks center distance or surface gap. For radius, mark center to rim. '
            'For diameter, mark opposite rim points. Preserve object color and instance identity. '
            'Do not invent hidden endpoints or convert pixels to physical units. If ambiguous/occluded, output {}. '
            + ('The metric scale is independently calibrated from the reference motion; return only target endpoints. '
               if motion_scale else 'Also mark endpoints of the stated reference dimension. If it is ruler calibre, mark two adjacent ruler ticks. ')
            + 'Output JSON only, coordinates normalized to 0..1000 relative to THIS supplied image: '
            + ('{"target":[[x1,y1],[x2,y2]]}.' if motion_scale else
               '{"reference":[[x1,y1],[x2,y2]],"target":[[x1,y1],[x2,y2]]}.'))
