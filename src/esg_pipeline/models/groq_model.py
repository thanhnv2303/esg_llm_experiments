from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import requests

from .base import ModelResponse, ModelRunner
from ._shared import normalise_label, retry_after_seconds, encode_image

LOGGER = logging.getLogger(__name__)


class GroqModel(ModelRunner):
    """Model runner for Groq's OpenAI-compatible chat completion API."""

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str = "https://api.groq.com/openai",
        name: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        timeout_seconds: int = 60,
        extra_headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        retry_wait_seconds: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.name = name or f"groq:{model}"
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
        content_chunks = [prompt.strip()]

        if page_text:
            content_chunks.append(f"\nExtracted page text:\n{page_text}")

        if page_image and page_image.exists():
            LOGGER.debug(
                "Groq chat completions currently do not support image inputs; ignoring %s",
                page_image,
            )

        combined_content = "\n\n".join(chunk for chunk in content_chunks if chunk)
        #
        payload: Dict[str, object] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": combined_content,
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
            LOGGER.debug("Sending Groq request to %s (attempt %s/%s)", url, attempt, self.max_retries)
            response = self.session.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout_seconds,
            )

            if response.status_code == 429 and attempt < self.max_retries:
                wait = retry_after_seconds(response, self.retry_wait_seconds)
                LOGGER.warning(
                    "Groq request hit rate limit; sleeping %.2f seconds before retry",
                    wait,
                )
                time.sleep(wait)
                continue

            try:
                response.raise_for_status()
                break
            except requests.HTTPError as exc:  # pragma: no cover - depends on network
                raise RuntimeError(
                    f"Groq model request failed with status {response.status_code}: {response.text}"
                ) from exc
        else:  # pragma: no cover - loop exhausted
            raise RuntimeError("Groq model request failed after maximum retries")

        assert response is not None
        data = response.json()
        LOGGER.debug("Received Groq response: %s", data)

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Groq model response did not contain any choices")

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
            "latency_ms": str(response.elapsed.total_seconds() * 1000),
        }
        usage = data.get("usage")
        if usage:
            metadata["usage"] = json.dumps(usage)
        x_groq = data.get("x_groq")
        if x_groq:
            metadata["x_groq"] = json.dumps(x_groq)

        return ModelResponse(
            label=label,
            raw_response=content_text,
            metadata=metadata,
        )


__all__ = ["GroqModel"]
