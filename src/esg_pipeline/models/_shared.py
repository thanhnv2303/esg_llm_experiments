from __future__ import annotations

import base64
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, Tuple, cast

import requests

from .base import PredictionLabel

_LABEL_MAP: Dict[str, PredictionLabel] = {
    "higher": "higher",
    "lower": "lower",
    "equal": "equal",
    "outperform": "higher",
    "underperform": "lower",
    "on": "equal",
    "on target": "equal",
    "not": "not found",
    "not found": "not found",
}


def encode_image(image_path: Path) -> str:
    """Return the inline-data representation expected by multimodal APIs."""

    with image_path.open("rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")
    suffix = image_path.suffix.lstrip(".") or "png"
    return f"data:image/{suffix};base64,{encoded}"


def encode_image_inline(image_path: Path) -> Tuple[str, str]:
    with image_path.open("rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")
    suffix = image_path.suffix.lstrip(".") or "png"
    return f"image/{suffix}", encoded
def _get_model_output(response_text: str):
    # Find the value for 'output'
    return re.search(r'output:\s*(.*)', response_text)

def normalise_label(text: str) -> PredictionLabel:
    if not text:
        return "not found"
    output = _get_model_output(text)
    if output in _LABEL_MAP:
        return cast(PredictionLabel, _LABEL_MAP[output])
    first = text.strip().split()[0].lower()
    if first in _LABEL_MAP:
        return cast(PredictionLabel, _LABEL_MAP[first])
    lowered = text.lower()
    for key, label in _LABEL_MAP.items():
        if key in lowered:
            return cast(PredictionLabel, label)
    return "not found"

def retry_after_seconds(response: requests.Response, default_wait: float) -> float:
    header = response.headers.get("Retry-After")
    if header:
        try:
            wait = float(header)
            if wait >= 0:
                return wait
        except ValueError:
            try:
                parsed = parsedate_to_datetime(header)
            except (TypeError, ValueError):
                parsed = None
            if parsed is not None:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                delta = (parsed - datetime.now(timezone.utc)).total_seconds()
                if delta > 0:
                    return delta
    return max(default_wait, 0.0)


__all__ = [
    "encode_image",
    "encode_image_inline",
    "normalise_label",
    "retry_after_seconds",
]
