from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional

PredictionLabel = Literal["higher", "lower", "equal", "not found"]


@dataclass
class ModelResponse:
    label: PredictionLabel
    raw_response: str
    metadata: Dict[str, str] = field(default_factory=dict)


class ModelRunner(ABC):
    name: str

    @abstractmethod
    def predict(
        self,
        prompt: str,
        page_image: Optional[Path] = None,
        page_text: Optional[str] = None,
    ) -> ModelResponse:
        """Run the model on the supplied prompt and page context."""


__all__ = ["PredictionLabel", "ModelResponse", "ModelRunner"]
