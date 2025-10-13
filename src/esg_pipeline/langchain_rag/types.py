from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .utils import serialise_json


@dataclass
class RetrievedChunk:
    """A lightweight representation of a retrieved document snippet."""

    content: str
    page: Optional[int]
    score: Optional[float]
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "content": self.content,
            "page": self.page,
        }
        if self.score is not None:
            payload["score"] = serialise_json(self.score)
        if self.metadata:
            payload["metadata"] = {
                key: serialise_json(value) for key, value in self.metadata.items()
            }
        return payload


__all__ = ["RetrievedChunk"]
