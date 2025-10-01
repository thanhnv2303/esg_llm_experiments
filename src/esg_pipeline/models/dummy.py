from __future__ import annotations

from typing import Optional

from .base import ModelResponse, ModelRunner, PredictionLabel


class DummyModel(ModelRunner):
    def __init__(self, name: str = "dummy", default_label: PredictionLabel = "not found") -> None:
        self.name = name
        self.default_label = default_label

    def predict(
        self,
        prompt: str,
        page_image: Optional[object] = None,
        page_text: Optional[str] = None,
    ) -> ModelResponse:
        response_text = (
            f"Model {self.name} received prompt of length {len(prompt)}. Returning default label {self.default_label}."
        )
        return ModelResponse(label=self.default_label, raw_response=response_text)


__all__ = ["DummyModel"]
