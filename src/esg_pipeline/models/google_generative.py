from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .base import ModelResponse, ModelRunner
from ._shared import encode_image_inline, normalise_label, retry_after_seconds

LOGGER = logging.getLogger(__name__)


class GoogleGenerativeModel(ModelRunner):
    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str = "https://generativelanguage.googleapis.com",
        name: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        timeout_seconds: int = 60,
        safety_settings: Optional[List[Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        retry_wait_seconds: float = 5.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.name = name or f"google-generative:{model}"
        self.temperature = temperature
        self.top_p = 0.95 if top_p is None else top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.safety_settings = safety_settings or []
        self.session = requests.Session()
        self.extra_headers = extra_headers or {}
        self.max_retries = max_retries
        self.retry_wait_seconds = retry_wait_seconds

    def predict(
        self,
        prompt: str,
        page_image: Optional[Path] = None,
        page_text: Optional[str] = None,
        page_images: Optional[List[Path]] = None,
    ) -> ModelResponse:
        parts: List[Dict[str, object]] = [{"text": prompt}]

        if page_text:
            parts.append({"text": f"\nExtracted page text:\n{page_text}"})

        image_candidates: List[Path] = []
        if page_image and page_image.exists():
            image_candidates.append(page_image)
        if page_images:
            for candidate in page_images:
                if candidate and candidate.exists():
                    image_candidates.append(candidate)

        unique_paths: List[Path] = []
        seen: set[Path] = set()
        for path in image_candidates:
            if path not in seen:
                unique_paths.append(path)
                seen.add(path)

        for path in unique_paths:
            try:
                mime_type, encoded = encode_image_inline(path)
                parts.append({"inline_data": {"mime_type": mime_type, "data": encoded}})
            except Exception as exc:  # pragma: no cover - depends on filesystem
                LOGGER.warning("Failed to encode image %s: %s", path, exc)

        payload: Dict[str, object] = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ]
        }

        generation_config: Dict[str, object] = {}
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        if self.top_p is not None:
            generation_config["topP"] = self.top_p
        if self.top_k is not None:
            generation_config["topK"] = self.top_k
        if self.max_output_tokens is not None:
            generation_config["maxOutputTokens"] = self.max_output_tokens
        if generation_config:
            payload["generationConfig"] = generation_config
        if self.safety_settings:
            payload["safetySettings"] = self.safety_settings

        url = f"{self.api_base}/v1beta/models/{self.model}:generateContent"

        headers = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        response: Optional[requests.Response] = None
        for attempt in range(1, self.max_retries + 1):
            LOGGER.debug(
                "Sending Google Generative AI request to %s (attempt %s/%s)",
                url,
                attempt,
                self.max_retries,
            )
            response = self.session.post(
                url,
                params={"key": self.api_key},
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout_seconds,
            )

            if response.status_code == 429 and attempt < self.max_retries:
                wait = retry_after_seconds(response, self.retry_wait_seconds)
                LOGGER.warning(
                    "Google Generative AI request hit rate limit; sleeping %.2f seconds before retry",
                    wait,
                )
                time.sleep(wait)
                continue

            try:
                response.raise_for_status()
                break
            except requests.HTTPError as exc:  # pragma: no cover - depends on network
                raise RuntimeError(
                    f"Model request failed with status {response.status_code}: {response.text}"
                ) from exc
        else:  # pragma: no cover - loop exhausted
            raise RuntimeError("Model request failed after maximum retries")

        assert response is not None
        data = response.json()
        LOGGER.debug("Received response: %s", data)

        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError("Model response did not contain any candidates")

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts_payload = content.get("parts") or []
        text_chunks = []
        for part in parts_payload:
            if isinstance(part, dict) and "text" in part:
                text_chunks.append(str(part["text"]))
        content_text = "".join(text_chunks)
        if not content_text:
            content_text = json.dumps(candidate)

        label = normalise_label(content_text)

        metadata: Dict[str, str] = {
            "model": self.model,
            "latency_ms": str(response.elapsed.total_seconds() * 1000),
        }
        if "usageMetadata" in candidate:
            metadata["usage"] = json.dumps(candidate["usageMetadata"])

        return ModelResponse(
            label=label,
            raw_response=content_text,
            metadata=metadata,
        )


__all__ = ["GoogleGenerativeModel"]
