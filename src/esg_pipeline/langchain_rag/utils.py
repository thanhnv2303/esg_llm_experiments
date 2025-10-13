from __future__ import annotations

import json
from typing import Iterable, Optional

import numpy as np

_VALUE_KEYS: Iterable[str] = (
    "extracted_value",
    "extractedValue",
    "extracted-value",
    "expected_value",
    "expectedValue",
    "expected-value",
    "company_value",
    "companyValue",
    "company-value",
)


def extract_extracted_value(raw_response: str) -> Optional[str]:
    if not raw_response:
        return None

    stripped = raw_response.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            for key in _VALUE_KEYS:
                if key in parsed and parsed[key] is not None:
                    return str(parsed[key]).strip(" ,")
    except json.JSONDecodeError:
        pass

    lower = stripped.lower()
    for key in _VALUE_KEYS:
        marker = f"{key}:"
        if marker in stripped:
            fragment = stripped.split(marker, 1)[-1]
            return fragment.splitlines()[0].strip().strip(",")
        marker = f"{key.replace('_', ' ').replace('-', ' ')}:"
        if marker in lower:
            fragment = lower.split(marker, 1)[-1]
            return fragment.splitlines()[0].strip().strip(",")
    return None


def serialise_json(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.integer):  # type: ignore[attr-defined]
        return int(value)
    if isinstance(value, np.floating):  # type: ignore[attr-defined]
        return float(value)
    return str(value)


__all__ = ["extract_extracted_value", "serialise_json"]
