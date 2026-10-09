"""A minimal chat-completion provider interface.

``OpenAICompatibleProvider`` speaks the ``/chat/completions`` protocol shared by OpenAI, Ollama,
vLLM and LM Studio. ``ScriptedProvider`` is a deterministic stand-in for tests and offline runs.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Protocol, TypedDict


class Message(TypedDict):
    role: str
    content: str


class ChatProvider(Protocol):
    name: str

    def complete(
        self, messages: list[Message], *, temperature: float = 0.0, max_tokens: int = 512
    ) -> str: ...


class ProviderError(RuntimeError):
    pass


class OpenAICompatibleProvider:
    """POSTs to ``{base_url}/chat/completions``; retries on 429 and 5xx with exponential backoff.

    The API key is read from ``FAITHGATE_LLM_API_KEY`` when not passed; local servers such as
    Ollama ignore it.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"openai-compatible:{model}"
        self.api_key = api_key if api_key is not None else os.environ.get("FAITHGATE_LLM_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(
        self, messages: list[Message], *, temperature: float = 0.0, max_tokens: int = 512
    ) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return str(payload["choices"][0]["message"]["content"])
            except urllib.error.HTTPError as err:
                retryable = err.code == 429 or err.code >= 500
                if not retryable or attempt == self.max_retries:
                    raise ProviderError(f"{self.base_url} answered HTTP {err.code}") from err
            except (urllib.error.URLError, TimeoutError) as err:
                if attempt == self.max_retries:
                    raise ProviderError(f"{self.base_url} unreachable: {err}") from err
            time.sleep(delay)
            delay *= 2
        raise ProviderError("unreachable")


class ScriptedProvider:
    """Answers with ``rule(messages)``; records every call so tests can inspect prompts."""

    def __init__(self, rule: Callable[[list[Message]], str], name: str = "scripted") -> None:
        self.rule = rule
        self.name = name
        self.calls: list[list[Message]] = []

    def complete(
        self, messages: list[Message], *, temperature: float = 0.0, max_tokens: int = 512
    ) -> str:
        self.calls.append(messages)
        return self.rule(messages)
