from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from .base import ModelResponse, ModelRunner, PredictionLabel
from ._shared import normalise_label
from ..rag_anything_pipeline import RAGAnythingPipeline, RAGAnythingPipelineConfig
from ..config import ExperimentConfig

LOGGER = logging.getLogger(__name__)


class RAGAnythingModel(ModelRunner):
    """Model runner that delegates predictions to the RAG-Anything pipeline."""

    name = "rag-anything"

    def __init__(
        self,
        config: RAGAnythingPipelineConfig,
        artifacts_dir: Optional[Path] = None,
    ) -> None:
        self.pipeline = RAGAnythingPipeline(config)
        self.artifacts_dir = artifacts_dir or Path("artifacts")
        self._prepared_document: Optional[Path] = None
        self.logger = LOGGER.getChild("RAGAnythingModel")

    def prepare_dataset(self, experiment: ExperimentConfig) -> None:
        """Process the experiment's PDF document through the RAG pipeline before querying."""

        document_path = experiment.dataset.document
        self.logger.info("Preparing document for RAG pipeline: %s", document_path)
        self.pipeline.prepare_document(document_path, experiment=experiment)
        self._prepared_document = document_path.resolve()

    def predict(
        self,
        prompt: str,
        page_image: Optional[Path] = None,
        page_text: Optional[str] = None,
        page_images: Optional[List[Path]] = None,
    ) -> ModelResponse:
        if self._prepared_document is None:
            raise RuntimeError(
                "RAGAnythingModel.prepare_dataset must be called before making predictions."
            )

        image_paths: List[Path] = []
        if page_image is not None:
            image_paths.append(page_image)
        if page_images:
            image_paths.extend(page_images)

        query_result = self.pipeline.query(prompt, image_paths=image_paths or None)
        raw_response = query_result.get("result", "") or ""
        label: PredictionLabel = normalise_label(raw_response)

        metadata = {
            "rag_mode": query_result.get("mode"),
            "rag_multimodal": query_result.get("multimodal", False),
            "rag_params": query_result.get("params", {}),
        }
        if self._prepared_document is not None:
            metadata["rag_document"] = str(self._prepared_document)

        return ModelResponse(
            label=label,
            raw_response=raw_response,
            metadata=metadata,
        )


__all__ = ["RAGAnythingModel"]
