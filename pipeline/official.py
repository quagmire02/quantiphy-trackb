"""Prompt construction and answer parsing copied from the official starter kit
(QuantiPhy/model_run_example/run_API_results.py) so VLM baselines match the paper's protocol.
Kept separate so the starter kit's import-time side effects (auto pip install) are avoided.
"""
import re

import numpy as np
import pandas as pd

SYSTEM_PROMPT = (
    "You are an expert video analyst specializing in physics measurements.\n"
    "Analyze the video frames carefully and provide ONLY the numerical answer with units. No explanation or reasoning needed.\n"
    "Format your response as: [value] [unit]\n"
    "Example: 2.5 cm\n"
    "Be as accurate as possible with measurements and calculations. Please give me an estimated answer even if you are not sure."
)


def build_context_prefix(prior=None, depth_info=None):
    parts = []
    if prior is not None and not pd.isna(prior):
        prior_text = str(prior).strip()
        if prior_text:
            parts.append(f"Given that {prior_text}.")
    if depth_info is not None and not pd.isna(depth_info):
        depth_text = str(depth_info).strip()
        if depth_text:
            parts.append(
                "Additionally, you have the following information about the distance between the objects in the video and the shooting camera: "
                f"{depth_text}"
            )
    if not parts:
        return ""
    context = " ".join(parts).strip()
    if context[-1] not in ".!?":
        context += "."
    return context + " "


def zero_shot_text(question, prior, video_type, depth_info):
    """User text for zero-shot v1/v3. Depth info is only included for 3D video types, as in the starter kit."""
    include_depth = len(str(video_type)) >= 2 and str(video_type)[1] == "3"
    prefix = build_context_prefix(prior, depth_info if include_depth else None)
    return f"{prefix}{question}\n\nPlease answer the question with numbers and units ONLY. No explanation needed."


num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
# Deliberate deviation from the starter kit: its pattern has an unescaped '^' in 'm/s^2', which
# never matches, so every "X m/s^2" reply fell through to the heuristic path and parsed as 2.0
# (the exponent). Escaping it makes such replies parse as X.
EXACT_NUMBER_WITH_UNIT_RE = re.compile(
    r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*(?:"
    r"meters?|meter|m|kilometers?|kilometer|km|centimeters?|centimeter|cm|millimeters?|millimeter|mm|"
    r"inches?|inch|in|feet|foot|ft|yards?|yard|yd|"
    r"seconds?|second|s|minutes?|minute|min|hours?|hour|h|"
    r"m/s|m/s²|m/s\^2|m/s2|km/h|kph|mph|ft/s|cm/s|mm/s|"
    r"m/s/s|m/s²|m/s\^2|m/s2|ft/s²|ft/s\^2|ft/s2|cm/s²|cm/s\^2|cm/s2|mm/s²|mm/s\^2|mm/s2|g"
    r")?\s*$",
    re.IGNORECASE,
)

_UNITS_TO_REMOVE = [
    'kilometers', 'kilometer', 'km', 'meters', 'meter', 'm', 'centimeters', 'centimeter', 'cm',
    'millimeters', 'millimeter', 'mm', 'inches', 'inch', 'in', 'feet', 'foot', 'ft', 'yards', 'yard', 'yd',
    'hours', 'hour', 'h', 'minutes', 'minute', 'min', 'seconds', 'second', 's',
    'km/h', 'kph', 'mph', 'ft/s', 'm/s', 'cm/s', 'mm/s',
    'm/s/s', 'm/s²', 'm/s^2', 'm/s2', 'ft/s²', 'ft/s^2', 'ft/s2', 'cm/s²', 'cm/s^2', 'cm/s2',
    'mm/s²', 'mm/s^2', 'mm/s2', 'g', 'm^2', 'm²',
]


def parse_number(text) -> float:
    """Official 5-step numeric parser (verbatim logic, except the '^' fix above)."""
    if not text or pd.isna(text):
        return np.nan
    original_text = str(text).strip()
    if EXACT_NUMBER_WITH_UNIT_RE.match(original_text):
        m = num_re.search(original_text)
        if m:
            try:
                return abs(float(m.group(0)))
            except ValueError:
                pass
    text_to_parse = original_text
    delimiters = ["Final Answer:", "Answer:", "=>", "=", ":", "is:", "The answer is:", "Example:"]
    last_pos, last_delim = -1, None
    for d in delimiters:
        positions = [m.start() for m in re.finditer(re.escape(d), text_to_parse, re.IGNORECASE)]
        if positions and max(positions) > last_pos:
            last_pos, last_delim = max(positions), d
    if last_pos >= 0 and last_delim:
        parts = text_to_parse.rsplit(last_delim, 1)
        if len(parts) > 1:
            text_to_parse = parts[-1].strip()
    cleaned = text_to_parse.lower()
    for unit in _UNITS_TO_REMOVE:
        cleaned = cleaned.replace(unit, '')
    for source in (cleaned, original_text):
        nums = num_re.findall(source)
        if nums:
            try:
                return abs(float(nums[-1]))
            except ValueError:
                pass
    return np.nan
