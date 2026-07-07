"""Small provider adapters for the configured model families."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from typing import Any

from agentic_kb.embeddings.models import EmbeddingModel
from agentic_kb.schemas.vectors import Embedding


class Qwen:
    """Qwen text-generation adapter.  Uses Qwen2.5-VL for vision-language tasks."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("api_key_env", "QWEN_API_KEY")
        kwargs.setdefault("base_url_env", "QWEN_BASE_URL")
        self._model = _ChatModel("Qwen2.5-VL-32B-Instruct", **kwargs)

    @property
    def model_name(self) -> str:
        return self._model.model_name

    def generate(self, prompt: str) -> str:
        return self._model.generate(prompt)

    def describe_image(self, image_bytes: bytes, prompt: str) -> str:
        return self._model.describe_image(image_bytes, prompt)


class DeepSeek:
    """DeepSeek text-generation adapter."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("api_key_env", "DEEPSEEK_API_KEY")
        kwargs.setdefault("base_url_env", "DEEPSEEK_BASE_URL")
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro").rsplit("/", 1)[-1]
        self._model = _ChatModel(model_name, **kwargs)

    @property
    def model_name(self) -> str:
        return self._model.model_name

    def generate(self, prompt: str) -> str:
        return self._model.generate(prompt)


class BgeEmbedding(EmbeddingModel):
    """BGE embedding adapter."""

    def __init__(
        self,
        *,
        dimensions: int = 1024,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "BGE_API_KEY",
        base_url_env: str = "BGE_BASE_URL",
        timeout: float | None = None,
        request_options: Mapping[str, Any] | None = None,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")

        self._model_name = os.getenv("BGE_MODEL", "bge-m3").rsplit("/", 1)[-1]
        self._dimensions = dimensions
        self._client = client or _client(api_key, base_url, api_key_env, base_url_env, timeout)
        self._options = dict(request_options or {})

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_texts(self, texts: list[str]) -> list[Embedding]:
        for index, text in enumerate(texts):
            if not text.strip():
                raise ValueError(f"text at index {index} is empty")

        response = self._client.embeddings.create(
            model=self._model_name,
            input=texts,
            **self._options,
        )
        return _embeddings(response)


class _ChatModel:
    def __init__(
        self,
        model_name: str,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "MODEL_API_KEY",
        base_url_env: str = "MODEL_BASE_URL",
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        extra_body: Mapping[str, Any] | None = None,
        request_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self._system_prompt = system_prompt
        self._client = client or _client(api_key, base_url, api_key_env, base_url_env, timeout)
        self._options = dict(request_options or {})
        if temperature is not None:
            self._options["temperature"] = temperature
        if max_tokens is not None:
            self._options["max_tokens"] = max_tokens
        if extra_body is not None:
            self._options["extra_body"] = dict(extra_body)

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self._options,
        )
        return _text(response)

    def describe_image(self, image_bytes: bytes, prompt: str) -> str:
        """Describe an image via the OpenAI vision API.  Retries up to 3 times
        when the model returns an empty or non-responsive answer."""
        import base64

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        })

        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **self._options,
                )
                result = _text(response).strip()
                if result:
                    return result
            except Exception as exc:
                last_error = exc
            if attempt < 3:
                import time
                time.sleep(1)

        if last_error is not None:
            raise last_error
        raise ValueError("image description returned empty response after 3 attempts")


def _client(
    api_key: str | None,
    base_url: str | None,
    api_key_env: str,
    base_url_env: str,
    timeout: float | None,
) -> Any:
    try:
        openai = importlib.import_module("openai")
    except ImportError as error:
        raise ImportError(
            "openai is required for provider adapters; install agentic-kb[providers]."
        ) from error

    options: dict[str, Any] = {}
    resolved_api_key = api_key if api_key is not None else os.getenv(api_key_env)
    resolved_base_url = base_url if base_url is not None else os.getenv(base_url_env)
    if resolved_api_key:
        options["api_key"] = resolved_api_key
    if resolved_base_url:
        options["base_url"] = resolved_base_url
    if timeout is not None:
        options["timeout"] = timeout
    return openai.OpenAI(**options)


def _text(response: Any) -> str:
    choices = _field(response, "choices") or []
    if not choices:
        raise ValueError("chat completion returned no choices")

    content = _field(_field(choices[0], "message"), "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [_field(part, "text") for part in content]
        text = "".join(part for part in text_parts if isinstance(part, str))
        if text:
            return text
    raise ValueError("chat completion returned no text content")


def _embeddings(response: Any) -> list[Embedding]:
    data = list(_field(response, "data") or [])
    if all(isinstance(_field(item, "index"), int) for item in data):
        data.sort(key=lambda item: _field(item, "index"))

    return [[float(value) for value in _field(item, "embedding")] for item in data]


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)
