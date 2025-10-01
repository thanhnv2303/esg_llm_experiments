from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import requests

from .base import ModelResponse, ModelRunner
from ._shared import encode_image, normalise_label, retry_after_seconds

LOGGER = logging.getLogger(__name__)


class OpenAICompatibleModel(ModelRunner):
    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        name: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        timeout_seconds: int = 60,
        extra_headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        retry_wait_seconds: float = 5.0,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.name = name or f"openai-compatible:{model}"
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.extra_headers = extra_headers or {}
        self.max_retries = max_retries
        self.retry_wait_seconds = retry_wait_seconds

    def predict(
        self,
        prompt: str,
        page_image: Optional[Path] = None,
        page_text: Optional[str] = None,
    ) -> ModelResponse:
        content = [{"type": "text", "text": prompt}]

        if page_text:
            content.append({"type": "text", "text": f"\nExtracted page text:\n{page_text}"})

        if page_image and page_image.exists():
            try:
                encoded = encode_image(page_image)
                content.append({"type": "image_url", "image_url": {"url": encoded}})
            except Exception as exc:  # pragma: no cover - depends on filesystem
                LOGGER.warning("Failed to encode image %s: %s", page_image, exc)

        payload: Dict[str, object] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        url = f"{self.api_base}/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        response: Optional[requests.Response] = None
        for attempt in range(1, self.max_retries + 1):
            LOGGER.debug("Sending request to %s (attempt %s/%s)", url, attempt, self.max_retries)
            response = self.session.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout_seconds,
            )
            if response.status_code == 429 and attempt < self.max_retries:
                wait = retry_after_seconds(response, self.retry_wait_seconds)
                LOGGER.warning(
                    "OpenAI-compatible request hit rate limit; sleeping %.2f seconds before retry",
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

        assert response is not None  # for type checkers
        data = response.json()
        LOGGER.debug("Received response: %s", data)

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Model response did not contain any choices")

        message = choices[0].get("message", {})
        content_text = message.get("content")
        if isinstance(content_text, list):
            # Some OpenAI-compatible servers return a list of content blocks
            content_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content_text
            )
        if not isinstance(content_text, str):
            content_text = json.dumps(content_text)

        label = normalise_label(content_text)

        return ModelResponse(
            label=label,
            raw_response=content_text,
            metadata={
                "model": self.model,
                "latency_ms": str(response.elapsed.total_seconds() * 1000),
            },
        )


__all__ = ["OpenAICompatibleModel"]
