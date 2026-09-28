"""Parse numbers/units from priors and questions; convert to and from SI."""
import re

import numpy as np

LENGTH = {"mm": 1e-3, "millimeter": 1e-3, "millimeters": 1e-3, "cm": 1e-2, "centimeter": 1e-2,
          "centimeters": 1e-2, "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
          "km": 1e3, "kilometer": 1e3, "kilometers": 1e3, "inch": 0.0254, "inches": 0.0254,
          "ft": 0.3048, "feet": 0.3048, "foot": 0.3048}
SPECIAL_SPEED = {"km/h": 1 / 3.6, "kph": 1 / 3.6, "mph": 0.44704}

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_UNIT_TOKEN = re.compile(  # lookbehind: a unit may follow a digit ("6m/s") but not a letter ("diameter")
    r"(?<![A-Za-z])(km/h|kph|mph|(?:mm|cm|km|m|ft)\s*/\s*s(?:\s*(?:\^\s*2|²|2|/\s*s))?|"
    r"millimeters?|centimeters?|kilometers?|meters?|metres?|mm|cm|km|inch(?:es)?|feet|foot|ft|m)\b",
    re.IGNORECASE)


def unit_factor(unit: str) -> tuple[float, int]:
    """'cm/s^2' -> (0.01, 2): SI factor and time power (0 length, 1 speed, 2 acceleration)."""
    u = unit.lower().replace(" ", "")
    if u in SPECIAL_SPEED:
        return SPECIAL_SPEED[u], 1
    if "/s" in u:
        base = u.split("/s")[0]
        power = 2 if re.search(r"(\^2|²|/s/s|s2)$", u) else 1
        return LENGTH[base], power
    return LENGTH[u], 0


def parse_prior_value(prior: str):
    """'speed of the bird =6m/s' -> (6.0, 'm/s'); unit may be None."""
    rhs = re.split(r"[=~≈]", str(prior))[-1]
    m = re.search(rf"({_NUM})\s*([a-zA-Z/^²2 ]*)", rhs)
    if not m:
        return None, None
    unit = _UNIT_TOKEN.search(m.group(2) or "")
    return float(m.group(1)), (unit.group(1) if unit else None)


def question_unit(question: str) -> str | None:
    """Requested output unit: the last unit token after the last ' in '."""
    q = str(question)
    tail = q[q.lower().rfind(" in ") + 4:] if " in " in q.lower() else q
    units = _UNIT_TOKEN.findall(tail)
    return units[-1] if units else None


def to_si(value: float, unit: str | None) -> float:
    """Convert to SI using only the unit's length scale; the dimension comes from the parsed
    quantity, so dataset typos like '9.8m/s' for an acceleration still convert correctly."""
    if unit is None:
        return value
    factor, _ = unit_factor(unit)
    return value * factor


def reply_to_unit(value: float, reply: str, target_unit: str | None) -> float:
    """A VLM reply like '1.5795 m' to a question asking for cm -> 157.95. Converts only when the
    reply states a unit of the same dimension (same time power); otherwise returns value unchanged."""
    units = _UNIT_TOKEN.findall(str(reply))
    if not units or not isinstance(target_unit, str) or not np.isfinite(value):
        return value
    try:
        (f_reply, p_reply), (f_target, p_target) = unit_factor(units[-1]), unit_factor(target_unit)
    except KeyError:
        return value
    return value * f_reply / f_target if p_reply == p_target else value


def from_si(value_si: float, unit: str | None) -> float:
    if unit is None:
        return value_si
    factor, _ = unit_factor(unit)
    return value_si / factor
