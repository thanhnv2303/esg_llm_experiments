from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .base import ModelResponse, ModelRunner
from ._shared import normalise_label, retry_after_seconds

LOGGER = logging.getLogger(__name__)


class CerebrasModel(ModelRunner):
    """Model runner for the Cerebras inference API (chat completions)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str = "https://api.cerebras.ai",
        name: Optional[str] = None,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: float = 60.0,
        extra_headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        retry_wait_seconds: float = 30.0,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.name = name or f"cerebras:{model}"
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_wait_seconds = retry_wait_seconds
        self.extra_headers = extra_headers or {}
        self.session = requests.Session()

    def _build_prompt(
        self,
        prompt: str,
        page_text: Optional[str],
        page_image: Optional[Path],
        page_images: Optional[List[Path]],
    ) -> str:
        prompt = prompt.strip()
        if page_text:
            prompt = f"{prompt}\n\nExtracted page text:\n{page_text}" if prompt else page_text
        if page_image and page_image.exists():
            LOGGER.warning(
                "Cerebras chat completions currently do not accept image inputs; ignoring %s",
                page_image,
            )
        if page_images:
            LOGGER.warning(
                "Cerebras chat completions currently do not accept image inputs; ignoring %d referenced images",
                len(page_images),
            )
        return prompt

    def predict(
        self,
        prompt: str,
        page_image: Optional[Path] = None,
        page_text: Optional[str] = None,
        page_images: Optional[List[Path]] = None,
    ) -> ModelResponse:
        user_prompt = self._build_prompt(prompt, page_text, page_image, page_images)

        payload: Dict[str, object] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        url = f"{self.api_base}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        response: Optional[requests.Response] = None
        latency_ms: float | None = None
        for attempt in range(1, self.max_retries + 1):
            LOGGER.debug(
                "Sending Cerebras request to %s (attempt %s/%s)",
                url,
                attempt,
                self.max_retries,
            )
            start = time.perf_counter()
            response = self.session.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout_seconds,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 429 and attempt < self.max_retries:
                wait = retry_after_seconds(response, self.retry_wait_seconds)
                LOGGER.warning(
                    "Cerebras request hit rate limit; sleeping %.2f seconds before retry",
                    wait,
                )
                time.sleep(wait)
                continue

            try:
                response.raise_for_status()
                break
            except requests.HTTPError as exc:  # pragma: no cover - network dependent
                raise RuntimeError(
                    f"Cerebras model request failed with status {response.status_code}: {response.text}"
                ) from exc
        else:  # pragma: no cover - loop exhausted
            raise RuntimeError("Cerebras model request failed after maximum retries")

        assert response is not None
        data = response.json()
        LOGGER.debug("Received Cerebras response: %s", data)

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Cerebras model response did not contain any choices")

        message = choices[0].get("message", {})
        content_text = message.get("content")
        if isinstance(content_text, list):
            content_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content_text
            )
        if not isinstance(content_text, str):
            content_text = json.dumps(content_text)

        label = normalise_label(content_text)

        metadata: Dict[str, str] = {
            "model": self.model,
        }
        if latency_ms is not None:
            metadata["latency_ms"] = f"{latency_ms:.2f}"
        usage = data.get("usage")
        if usage:
            metadata["usage"] = json.dumps(usage)

        return ModelResponse(
            label=label,
            raw_response=content_text,
            metadata=metadata,
        )


__all__ = ["CerebrasModel"]
