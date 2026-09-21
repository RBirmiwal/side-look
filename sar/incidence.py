"""Capella collect geometry: incidence from the embedded extended JSON.

The MAG-POL GeoTIFF stores the Capella product JSON in TIFFTAG_IMAGEDESCRIPTION.
The field is named incidence_angle and is measured from vertical (nadir = 0).
Grazing / elevation / depression would be from horizontal and need 90 - x.
"""

from __future__ import annotations

import json
from typing import Any

# Exact path in the Capella product JSON (the *_extended.json contents).
INCIDENCE_PATH = ("collect", "image", "center_pixel", "incidence_angle")
INCIDENCE_FIELD = "collect.image.center_pixel.incidence_angle"

_FROM_HORIZONTAL = ("grazing", "elevation", "depression")


def _dig(obj: Any, path: tuple[str, ...]) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(INCIDENCE_FIELD)
        cur = cur[key]
    return cur


def parse_incidence(meta: dict) -> dict:
    """Return field name, raw value, theta-from-vertical, and the 90-x pair.

    Fails loudly if the field is missing — a guessed angle silently poisons
    every mask in the dataset.
    """
    raw = _dig(meta, INCIDENCE_PATH)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{INCIDENCE_FIELD} is not a number: {raw!r}") from exc
    if not (0.0 < value < 90.0):
        raise ValueError(f"{INCIDENCE_FIELD} out of (0, 90): {value}")

    leaf = INCIDENCE_PATH[-1].lower()
    from_horizontal = any(token in leaf for token in _FROM_HORIZONTAL)
    if from_horizontal:
        theta = 90.0 - value
        convention = "horizontal (converted with 90 - x)"
    else:
        theta = value
        convention = "vertical (field is named incidence, not grazing)"

    return {
        "field": INCIDENCE_FIELD,
        "raw_value_deg": value,
        "theta_from_vertical_deg": theta,
        "ninety_minus_raw_deg": 90.0 - value,
        "convention": convention,
        "converted": from_horizontal,
    }


def incidence_from_tags(tags: dict) -> dict:
    raw = tags.get("TIFFTAG_IMAGEDESCRIPTION") or tags.get("IMAGEDESCRIPTION")
    if not raw:
        raise ValueError("TIFF has no IMAGEDESCRIPTION; cannot read incidence")
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("IMAGEDESCRIPTION is not Capella JSON") from exc
    return parse_incidence(meta)
