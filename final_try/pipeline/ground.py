"""Open-vocabulary grounding with Qwen3-VL: object phrase + frame -> bounding box.

Qwen3-VL emits boxes as JSON with coordinates normalised to 0..1000; we convert to pixels
of the frame we passed in.
"""
import json
import re

import numpy as np
from PIL import Image

PROMPT = ('Locate the {obj} in the image. Coordinates must be normalized to 0..1000. Output its bounding box as JSON: '
          '[{{"bbox_2d": [x1, y1, x2, y2], "label": "{obj}"}}]. '
          'If it is not visible, output [].')
PROMPT_ALL = ('Locate every {obj} in the image. Coordinates must be normalized to 0..1000. Output one bounding box per {obj} as a JSON list: '
              '[{{"bbox_2d": [x1, y1, x2, y2], "label": "{obj}"}}, ...]. If none is visible, output [].')


_QUAD = re.compile(r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]")


def parse_boxes(reply: str) -> list[list[float]]:
    """All [x1, y1, x2, y2] quadruples in the reply (works on truncated JSON too)."""
    boxes = [[float(v) for v in m] for m in _QUAD.findall(reply)]
    # Keep the established truncated-JSON path, and support named coordinate objects.
    for start, char in enumerate(reply):
        if char not in '[{':
            continue
        try:
            obj, _ = json.JSONDecoder().raw_decode(reply[start:])
        except (ValueError, TypeError):
            continue
        def visit(value):
            if isinstance(value, dict):
                if all(k in value for k in ('x1', 'y1', 'x2', 'y2')):
                    boxes.append([value[k] for k in ('x1', 'y1', 'x2', 'y2')])
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(obj)
    out = []
    for box in boxes:
        try:
            b = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        if all(np.isfinite(v) and 0 <= v <= 1000 for v in b) and b not in out:
            out.append(b)
    return out


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


_AREA = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
_PICK = [  # contrastive modifier -> how to choose among candidate boxes
    (r"\b(big|bigger|biggest|large|larger|largest)\b", _AREA, max),
    (r"\b(small|smaller|smallest|little|tiny)\b", _AREA, min),
    (r"\bleft\b", lambda b: b[0] + b[2], min),
    (r"\bright\b", lambda b: b[0] + b[2], max),
    (r"\b(top|upper|higher)\b", lambda b: b[1] + b[3], min),
    (r"\b(bottom|lower)\b", lambda b: b[1] + b[3], max),
    # depth words: farther objects look smaller
    (r"\b(back|behind|far|farther|further|rear|distant|background)\b", _AREA, min),
    (r"\b(front|near|nearer|closer|foreground)\b", _AREA, max),
]


def pick_by_modifier(phrase: str, boxes: list) -> list | None:
    for pat, key, choose in _PICK:
        if re.search(pat, phrase.lower()):
            return choose(boxes, key=key)
    return None


# Colour words -> HSV rule (OpenCV: H 0-179, S and V 0-255), evaluated per pixel.
_COLOURS = {
    "black": lambda h, s, v: v < 60,
    "white": lambda h, s, v: (s < 45) & (v > 175),
    "grey": lambda h, s, v: (s < 45) & (v >= 60) & (v <= 200),
    "red": lambda h, s, v: ((h < 8) | (h > 170)) & (s > 90) & (v > 60),
    "orange": lambda h, s, v: (h >= 8) & (h < 20) & (s > 90) & (v > 170),  # brighter than brown
    "brown": lambda h, s, v: (h >= 5) & (h < 25) & (s > 60) & (v > 30) & (v <= 170),
    "yellow": lambda h, s, v: (h >= 20) & (h < 35) & (s > 80) & (v > 90),
    "green": lambda h, s, v: (h >= 35) & (h < 85) & (s > 50) & (v > 40),
    "blue": lambda h, s, v: (h >= 85) & (h < 130) & (s > 50) & (v > 40),
    "purple": lambda h, s, v: (h >= 130) & (h < 155) & (s > 50) & (v > 40),
    "pink": lambda h, s, v: (((h >= 145) & (h <= 179)) | (h < 8)) & (s > 30) & (s < 160) & (v > 150),
}
_COLOUR_WORDS = {"gray": "grey", "silver": "grey", "violet": "purple", "golden": "yellow", "gold": "yellow"}


def colour_word(phrase: str) -> str | None:
    for w in re.findall(r"[a-z]+", phrase.lower()):
        w = _COLOUR_WORDS.get(w, w)
        if w in _COLOURS:
            return w
    return None


def colour_fraction(frame: np.ndarray, box, colour: str) -> float:
    """Share of pixels in the central part of the box that look like `colour`."""
    import cv2
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * 0.2, (y2 - y1) * 0.2
    crop = frame[int(y1 + dy):int(max(y1 + dy + 1, y2 - dy)), int(x1 + dx):int(max(x1 + dx + 1, x2 - dx))]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(int)
    return float(_COLOURS[colour](hsv[:, 0], hsv[:, 1], hsv[:, 2]).mean())


def pick_by_colour(phrase: str, boxes: list, frame: np.ndarray, margin: float = 0.15) -> list | None:
    """Candidate that best matches the phrase's colour word, if it clearly beats the others."""
    colour = colour_word(phrase)
    if colour is None or len(boxes) < 2:
        return None
    scores = [colour_fraction(frame, b, colour) for b in boxes]
    order = np.argsort(scores)[::-1]
    return boxes[order[0]] if scores[order[0]] - scores[order[1]] >= margin else None


def ground(vlm, frame: np.ndarray, obj: str, every: bool = False) -> dict:
    """Return {'boxes': [[x1,y1,x2,y2], ...] in frame pixels (Qwen's order), 'reply': raw}.
    every=True asks for all instances instead of 'the' object."""
    h, w = frame.shape[:2]
    prompt = (PROMPT_ALL if every else PROMPT).format(obj=obj)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    reply = vlm.chat(messages, images=[Image.fromarray(frame)], max_new_tokens=256)
    if hasattr(vlm, 'row') and parse_boxes(reply):
        # Context-wrapped recovery calls also verify against the same image/question.
        verify = (prompt + '\nUnverified candidate boxes: ' + json.dumps(parse_boxes(reply))
                  + '\nCheck identity and full silhouette in the image. Correct the coordinates, '
                    'or output [] if the requested object is not visible. Return the requested JSON only.')
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': verify}]}]
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


def simple_phrase(phrase: str) -> str:
    """Plain noun phrase a small model grounds more easily: drop location clauses and parentheses,
    fix common typos ('bycicle on the upper left conrner' -> 'bicycle')."""
    p = re.sub(r"\([^)]*\)", " ", phrase.lower().split("#")[0])
    words = p.split()
    for i, w in enumerate(words):
        if w in _PREPOSITIONS and i > 0:
            words = words[:i]
            break
    fixes = {"bycicle": "bicycle", "bicyle": "bicycle", "ballon": "balloon", "soccor": "soccer", "chess": "chess piece"}
    return " ".join(fixes.get(w, w) for w in words).strip()


def ground_tiled(vlm, frame: np.ndarray, phrase: str, zoom: float = 2.0) -> dict:
    """Ground on 2x2 overlapping tiles, each upscaled `zoom` times, so a small object covers more of
    the VLM's view. Returns the first tile hit (box mapped back to frame pixels), else no boxes."""
    import cv2
    H, W = frame.shape[:2]
    th, tw = int(H * 0.6), int(W * 0.6)
    for y0 in (0, H - th):
        for x0 in (0, W - tw):
            tile = cv2.resize(frame[y0:y0 + th, x0:x0 + tw], (int(tw * zoom), int(th * zoom)),
                              interpolation=cv2.INTER_CUBIC)
            g = ground(vlm, tile, phrase)
            for b in g["boxes"]:
                box = [b[0] / zoom + x0, b[1] / zoom + y0, b[2] / zoom + x0, b[3] / zoom + y0]
                # a box hugging the tile edge is probably a cut-off object: let another tile take it
                if box[2] - box[0] < 0.9 * tw and box[3] - box[1] < 0.9 * th:
                    return {"boxes": [[round(v, 1) for v in box]], "reply": g["reply"], "tile": [x0, y0]}
    return {"boxes": [], "reply": ""}


def _words(phrase: str) -> set[str]:
    return set(phrase.lower().split("#")[0].split()) - {"the", "of", "a", "an"}


def resolve_conflicts(vlm, frames: np.ndarray, objects: dict) -> dict:
    """Different objects with the same head noun ('big bird' / 'small bird') that received
    overlapping boxes on the same frame: re-ground every instance of the bare noun and assign
    candidates by the contrastive modifier (size, left-right, top-bottom, front-back), else by colour
    word, else by elimination. If only one such object exists in the clip, the two phrases describe
    the same object and keep the shared box. Unresolvable ones lose their box (-> VLM fallback)
    instead of silently sharing one mask.
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
            if len(cands) <= 1:
                # A missed second object is not proof these distinct phrases are the same.
                for k in (a, b):
                    objects[k]["conflict"] = {"with": b if k == a else a, "resolved": False,
                                              "candidates": cands}
                    objects[k]['boxes'] = []
                continue
            phrases = {k: objects[k].get("phrase", k) for k in (a, b)}
            picks = {k: pick_by_modifier(phrases[k], cands) or pick_by_colour(phrases[k], cands, frames[fi])
                     for k in (a, b)}
            # Elimination: with two candidates, a resolved phrase leaves the other candidate.
            if len(cands) == 2 and sum(v is not None for v in picks.values()) == 1:
                known = next(k for k in (a, b) if picks[k] is not None)
                other = next(k for k in (a, b) if k != known)
                picks[other] = next(c for c in cands if c is not picks[known])
            ok = all(v is not None for v in picks.values()) and iou(picks[a], picks[b]) < 0.5
            for k in (a, b):
                objects[k]["conflict"] = {"with": b if k == a else a, "resolved": ok, "candidates": cands,
                                          "frame": fi}
                objects[k]["boxes"] = [picks[k]] if ok else []
                if ok:
                    objects[k]["frame"] = fi
    return objects
