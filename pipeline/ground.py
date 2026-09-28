"""Open-vocabulary grounding with Qwen3-VL: object phrase + frame -> bounding box.

Qwen3-VL emits boxes as JSON with coordinates normalised to 0..1000; we convert to pixels
of the frame we passed in.
"""
import re

import numpy as np
from PIL import Image

PROMPT = ('Locate the {obj} in the image. Output its bounding box as JSON: '
          '[{{"bbox_2d": [x1, y1, x2, y2], "label": "{obj}"}}]. '
          'If it is not visible, output [].')
PROMPT_ALL = ('Locate every {obj} in the image. Output one bounding box per {obj} as a JSON list: '
              '[{{"bbox_2d": [x1, y1, x2, y2], "label": "{obj}"}}, ...]. If none is visible, output [].')


_QUAD = re.compile(r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]")


def parse_boxes(reply: str) -> list[list[float]]:
    """All [x1, y1, x2, y2] quadruples in the reply (works on truncated JSON too)."""
    return [[float(v) for v in m] for m in _QUAD.findall(reply)]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


_PREPOSITIONS = {"in", "on", "at", "of", "with", "near", "under", "above", "below", "behind", "from", "to"}


def head_noun(phrase: str) -> str:
    """'person in the black shirt' -> 'person'; 'big bird' -> 'bird'."""
    words = phrase.lower().split("#")[0].split()
    for i, w in enumerate(words):
        if w in _PREPOSITIONS and i > 0:
            words = words[:i]
            break
    return words[-1] if words else ""


_PICK = [  # contrastive modifier -> how to choose among candidate boxes
    (r"\b(big|bigger|biggest|large|larger|largest)\b", lambda b: (b[2] - b[0]) * (b[3] - b[1]), max),
    (r"\b(small|smaller|smallest|little|tiny)\b", lambda b: (b[2] - b[0]) * (b[3] - b[1]), min),
    (r"\bleft\b", lambda b: b[0] + b[2], min),
    (r"\bright\b", lambda b: b[0] + b[2], max),
    (r"\b(top|upper|higher)\b", lambda b: b[1] + b[3], min),
    (r"\b(bottom|lower)\b", lambda b: b[1] + b[3], max),
]


def pick_by_modifier(phrase: str, boxes: list) -> list | None:
    for pat, key, choose in _PICK:
        if re.search(pat, phrase.lower()):
            return choose(boxes, key=key)
    return None


def ground(vlm, frame: np.ndarray, obj: str, every: bool = False) -> dict:
    """Return {'boxes': [[x1,y1,x2,y2], ...] in frame pixels (Qwen's order), 'reply': raw}.
    every=True asks for all instances instead of 'the' object."""
    h, w = frame.shape[:2]
    prompt = (PROMPT_ALL if every else PROMPT).format(obj=obj)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    reply = vlm.chat(messages, images=[Image.fromarray(frame)], max_new_tokens=256)
    boxes = []
    for x1, y1, x2, y2 in parse_boxes(reply):
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        box = [round(v, 1) for v in (x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h)]
        # Drop degenerate boxes and near-duplicates of an earlier candidate.
        if box[2] - box[0] >= 2 and box[3] - box[1] >= 2 and all(iou(box, b) < 0.7 for b in boxes):
            boxes.append(box)
    return {"boxes": boxes, "reply": reply}


def _words(phrase: str) -> set[str]:
    return set(phrase.lower().split("#")[0].split()) - {"the", "of", "a", "an"}


def resolve_conflicts(vlm, frames: np.ndarray, objects: dict) -> dict:
    """Different objects with the same head noun ('big bird' / 'small bird') that received
    overlapping boxes on the same frame: re-ground every instance of the bare noun and assign
    candidates by the contrastive modifier (size / left-right / top-bottom). Unresolvable ones
    lose their box (-> VLM fallback) instead of silently sharing one mask.
    Phrases whose words are a subset of the other's ('bird' / 'big bird') may be the same object
    and are left alone."""
    bases = [b for b, o in objects.items() if o["boxes"]]
    for i, a in enumerate(bases):
        for b in bases[i + 1:]:
            oa, ob = objects[a], objects[b]
            if not (oa["boxes"] and ob["boxes"]) or oa["frame"] != ob["frame"]:
                continue
            pa, pb = oa.get("phrase", a), ob.get("phrase", b)
            wa, wb = _words(pa), _words(pb)
            if head_noun(pa) != head_noun(pb) or wa <= wb or wb <= wa:
                continue
            if iou(oa["boxes"][0], ob["boxes"][0]) < 0.5:
                continue
            # Both instances may not be visible at the key frame (e.g. one flew out): search a few frames.
            n = len(frames)
            for fi in dict.fromkeys([oa["frame"], 0, n // 4, n // 2, 3 * n // 4]):
                cands = ground(vlm, frames[fi], head_noun(pa), every=True)["boxes"]
                if len(cands) >= 2:
                    break
            picks = {k: pick_by_modifier(objects[k].get("phrase", k), cands) if len(cands) >= 2 else None
                     for k in (a, b)}
            ok = all(picks.values()) and iou(picks[a], picks[b]) < 0.5
            for k in (a, b):
                objects[k]["conflict"] = {"with": b if k == a else a, "resolved": ok, "candidates": cands,
                                          "frame": fi}
                objects[k]["boxes"] = [picks[k]] if ok else []
                if ok:
                    objects[k]["frame"] = fi
    return objects
