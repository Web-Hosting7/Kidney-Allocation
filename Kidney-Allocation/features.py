"""
Loads features.json once and exposes it in the shapes the rest of the app
needs. To swap in a different set of study features, edit features.json —
nothing in this file (or anywhere else) needs to change as long as each
feature entry has the same fields.

FEATURE_KEYS     ["dependents", "age", ...]                 — study order
FEATURE_BY_KEY   {"age": {...the full JSON entry...}, ...}
"""

import json
import os

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.json")


def _load():
    with open(_PATH) as f:
        data = json.load(f)
    feats = data["features"]
    for f_ in feats:
        missing = [k for k in ("key", "label", "description", "min", "max",
                                "gap_phrase", "direction_higher", "direction_lower",
                                "direction_higher_short", "direction_lower_short")
                   if k not in f_]
        if missing:
            raise ValueError(f"features.json entry {f_.get('key', '?')!r} is missing: {missing}")
    return feats


FEATURES_DATA = _load()
FEATURE_KEYS = [f["key"] for f in FEATURES_DATA]
FEATURE_BY_KEY = {f["key"]: f for f in FEATURES_DATA}


def descriptions():
    """key -> description, for the start page."""
    return {k: v["description"] for k, v in FEATURE_BY_KEY.items()}


def ranges():
    """key -> (min, max), for scenario generation and sliders."""
    return {k: (v["min"], v["max"]) for k, v in FEATURE_BY_KEY.items()}


def direction_labels(short=False):
    """key -> (label when higher, label when lower)."""
    suffix = "_short" if short else ""
    return {k: (v[f"direction_higher{suffix}"], v[f"direction_lower{suffix}"])
            for k, v in FEATURE_BY_KEY.items()}


def gap_phrases():
    """key -> (gap phrase, unit) — e.g. ("age gap", "years"), used in captions."""
    return {k: (v["gap_phrase"], v.get("unit", "")) for k, v in FEATURE_BY_KEY.items()}