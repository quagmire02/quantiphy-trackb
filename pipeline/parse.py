"""Turn (prior, question) into a structured measurement request.

Prior (once per video, so every question of a clip shares one reference object): object,
quantity, time, value and unit come from rules on the prior text; the local VLM (text-only)
is asked only when the prior names no object ("gravity acc", "walking velocity").
Question (once per item): the VLM extracts target object(s), quantity and time; unit by regex.

Few-shot examples are invented, not taken from the validation or test sets.
"""
import json
import re

from pipeline.units import parse_prior_value, question_unit

SIZE_Q = ["length", "width", "height", "diameter", "radius", "thickness", "size"]
MOTION_Q = ["speed", "acceleration"]
TARGET_Q = SIZE_Q + MOTION_Q + ["distance", "displacement", "path_length", "path_diameter"]

TIME_RULES = ('Times are [start, end] in seconds; [t, t] for an instant; -1 means the end of the video; '
              'null when no time is stated. "before 0.6s" -> [0, 0.6]; "after 0.75s" -> [0.75, -1]; '
              '"initial" -> [0, 0]; "final" -> [-1, -1].')

PRIOR_SYSTEM = f"""You read the physical prior given for a short video and output JSON only.
Keys:
"ref_object": the object the prior is about, as a short phrase to find it in the video (e.g. "boat", "red ball").
  If the prior is gravity or an acceleration without a named object, use the object that falls or flies freely (use the questions to decide).
  If the prior names no object (e.g. "walking velocity"), use the object it plausibly describes (e.g. "walking person").
"ref_time": when the prior applies. {TIME_RULES}"""

PRIOR_EXAMPLES = [
    ("gravity acceleration = 9.8 m/s^2", ["What is the diameter of the falling apple in cm?", "What is the height of the table in cm?"],
     {"ref_object": "falling apple", "ref_time": None}),
    ("jogging speed = 2.5 m/s", ["What is the length of the bench in meters?", "What is the height of the runner in meters?"],
     {"ref_object": "jogging person", "ref_time": None}),
]

QUESTION_SYSTEM = f"""You read a physics question about a short video and output JSON only.
Keys:
"targets": the object(s) asked about, as short singular phrases. Two objects only for a distance between two objects; for "the two X" repeat the phrase twice.
"target_quantity": one of {json.dumps(TARGET_Q)}.
  "speed" for speed or velocity; "path_length" for total distance travelled / flight path length;
  "path_diameter" for the diameter of an orbit or circular path; "displacement" for net change of position.
"target_time": {TIME_RULES} Use null for static sizes and whole-video averages."""

QUESTION_EXAMPLES = [
    ("What is the width of the red kayak in meters?",
     {"targets": ["red kayak"], "target_quantity": "width", "target_time": None}),
    ("What is the velocity of the frisbee at 0.8s in m/s?",
     {"targets": ["frisbee"], "target_quantity": "speed", "target_time": [0.8, 0.8]}),
    ("What is the distance between the two traffic cones in meters?",
     {"targets": ["traffic cone", "traffic cone"], "target_quantity": "distance", "target_time": None}),
    ("What is the average speed of the green marble between 0.5s and 1.5s in cm/s?",
     {"targets": ["green marble"], "target_quantity": "speed", "target_time": [0.5, 1.5]}),
    ("What is the diameter of the circular path of the toy train?",
     {"targets": ["toy train"], "target_quantity": "path_diameter", "target_time": None}),
]


def _chat(system, examples, user):
    msgs = [{"role": "system", "content": [{"type": "text", "text": system}]}]
    for u, out in examples:
        msgs.append({"role": "user", "content": [{"type": "text", "text": u}]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": json.dumps(out)}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": user}]})
    return msgs


def _prior_user(prior, questions):
    return f'Prior: "{prior}"\nQuestions about this video:\n' + "\n".join(f"- {q}" for q in questions)


def prior_messages(prior: str, questions: list[str]) -> list[dict]:
    return _chat(PRIOR_SYSTEM, [(_prior_user(p, qs), o) for p, qs, o in PRIOR_EXAMPLES], _prior_user(prior, questions))


def question_messages(question: str) -> list[dict]:
    return _chat(QUESTION_SYSTEM, [(f'Question: "{q}"', o) for q, o in QUESTION_EXAMPLES], f'Question: "{question}"')


def _json(reply: str) -> dict:
    m = re.search(r"\{.*\}", reply, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def _norm_time(t):
    if isinstance(t, (int, float)):
        return [float(t), float(t)]
    if isinstance(t, list) and len(t) == 2 and all(isinstance(x, (int, float)) or x is None for x in t):
        return [0.0 if t[0] is None else float(t[0]), -1.0 if t[1] is None else float(t[1])]
    return None


_NUM = r"\d*\.?\d+"
_TIME_PATTERNS = [  # (regex on the prior description, -> [start, end])
    (rf"^\s*t\s*=\s*({_NUM})\s*s?\s*,", lambda m: [float(m[1]), float(m[1])]),
    (rf"\bfrom\s+({_NUM})\s*s?\s+to\s+({_NUM})\s*s\b", lambda m: [float(m[1]), float(m[2])]),
    (rf"\bbefore\s+({_NUM})\s*s\b", lambda m: [0.0, float(m[1])]),
    (rf"\bafter\s+({_NUM})\s*s\b", lambda m: [float(m[1]), -1.0]),
    (rf"\bbetween\s+({_NUM})\s*s?\s+and\s+({_NUM})\s*s\b", lambda m: [float(m[1]), float(m[2])]),
    (rf"\b(?:at|when\s+t\s*=)\s*(?:t\s*=\s*)?({_NUM})\s*s\b", lambda m: [float(m[1]), float(m[1])]),
]
_QUANTITY_WORDS = r"\b(average|mean|lifting|walking|running|orbital|speed|velocity|acceleration|acc|gravity|" \
                  r"diameter|radius|length|lenth|width|height|thickness|size|calibre|caliber|the|a|an)\b"


def first_prior(prior) -> str:
    """A few priors hold two statements on separate lines; use the first consistently."""
    return str(prior).strip().splitlines()[0].strip()


def prior_description(prior: str) -> str:
    """Left-hand side of the prior, e.g. 't=0.6, ball acceleration = 4.6m/s^2' -> 't=0.6, ball acceleration'."""
    m = re.match(r"^(.*)[=~≈]\s*[-+]?\d*\.?\d", first_prior(prior))  # greedy: last '=' before the value
    return (m.group(1) if m else first_prior(prior)).replace("_", " ").strip()


def prior_time(desc: str):
    for pat, fn in _TIME_PATTERNS:
        m = re.search(pat, desc, re.I)
        if m:
            return fn(m)
    return None


def prior_quantity(desc: str) -> str | None:
    d = desc.lower().replace("lenth", "length")
    if re.search(r"accel|gravity|\bacc\b", d):
        return "acceleration"
    if re.search(r"\b(speed|velocity)\b", d):  # not "speedboat"
        return "speed"
    for q in ["diameter", "radius", "length", "width", "height", "thickness"]:
        if re.search(rf"\b{q}\b", d):
            return q
    return "size" if re.search(r"size|long|tall|wide", d) else None


def prior_object(desc: str) -> str:
    """Object named in the prior by rule, or '' if it names none (then the VLM decides).
    'speed of the shark' -> 'shark'; 'credit card width' -> 'credit card';
    'length of the raccoon (tail included)' -> 'raccoon'; 'gravity acc' -> ''."""
    d = re.sub(r"\([^)]*\)", " ", desc.lower())
    for pat, _ in _TIME_PATTERNS:
        d = re.sub(pat, " ", d, flags=re.I)
    d = re.sub(r"\b(at|before|after|when)\b\s*$", " ", d)
    if " of " in d:
        d = d.split(" of ", 1)[1]
    d = re.sub(r"'s\b", "", d) if re.search(r"'s\s+(model\s+)?(speed|velocity|size|length|height|diameter)", d) else d
    d = re.sub(_QUANTITY_WORDS, " ", d)
    return " ".join(re.sub(r"[^a-z0-9' -]", " ", d).split())


def finalize_prior(reply: str, prior: str) -> dict:
    """reply: VLM answer to prior_messages, used only if the prior names no object ('' allowed)."""
    desc = prior_description(prior)
    value, unit = parse_prior_value(first_prior(prior))
    rule_obj = prior_object(desc)
    llm = _json(reply)
    rec = {"ref_object": rule_obj or str(llm.get("ref_object") or "").strip(),
           "ref_object_source": "rule" if rule_obj else "vlm",
           "ref_quantity": prior_quantity(desc),
           "ref_time": prior_time(desc) if rule_obj or prior_time(desc) else _norm_time(llm.get("ref_time")),
           "prior_value": value, "prior_unit": unit, "prior_raw": reply}
    if rec["ref_quantity"] == "radius":  # measure diameters, halve the prior's scale
        rec["ref_quantity"], rec["prior_value"] = "diameter", None if value is None else 2 * value
    # Ruler-tick priors ("ruler calibre = 1 cm") need tick-spacing detection; not supported yet.
    unsupported = bool(re.search(r"ruler|calibre|caliber", str(prior), re.I))
    rec["prior_ok"] = bool(not unsupported and rec["ref_object"] and rec["prior_value"] is not None
                           and rec["ref_quantity"])
    return rec


_COMPONENT = re.compile(r"\b(horizontal|vertical)\w*\b(?:\s*\([^)]*\))?\s*(?:\w+\s+){0,2}?"
                        r"(velocity|speed|displacement|distance)|\b(velocity|speed|displacement)\b[^?]*\b(horizontal|vertical)\w*\b", re.I)


def question_overrides(question: str) -> dict:
    """Meaning the generic parse misses, by rule on the question text (applied at load time):
    'horizontal speed / vertical displacement' -> only that image axis of the motion (2D);
    'wingspan' -> the long axis of the mask, whatever dimension word the question uses."""
    out = {}
    m = _COMPONENT.search(str(question))
    # Only one object's own motion: "horizontal distance between two cars" means along the ground,
    # not along the image x axis.
    if m and not re.search(r"\bbetween\b|\bfrom\b.*\bto\b", str(question), re.I):
        out["component"] = (m.group(1) or m.group(4)).lower()
    if re.search(r"wing\s*span", str(question), re.I):
        out["target_quantity"] = "length"
    return out


def finalize_question(reply: str, question: str) -> dict:
    js = _json(reply)
    targets = [str(t).strip() for t in js.get("targets") or [] if str(t).strip()][:2]
    q = js.get("target_quantity")
    ql = question.lower()
    if "acceleration" in ql:
        q = "acceleration"
    elif re.search(r"\b(speed|velocity)\b", ql) and q != "speed":  # not "speedboat"
        q = "speed"
    rec = {"targets": targets, "target_quantity": q, "target_time": _norm_time(js.get("target_time")),
           "target_unit": question_unit(question), "question_raw": reply}
    rec["question_ok"] = bool(targets and q in TARGET_Q and (q != "distance" or len(targets) == 2))
    return rec
