from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from faithgate.llm import OpenAICompatibleProvider, ProviderError


class _Handler(BaseHTTPRequestHandler):
    statuses: ClassVar[list[int]] = []
    seen: ClassVar[list[dict[str, object]]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.seen.append(
            {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
        )
        status = self.statuses.pop(0) if self.statuses else 200
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if status == 200:
            reply = {"choices": [{"message": {"role": "assistant", "content": "pong"}}]}
            self.wfile.write(json.dumps(reply).encode())

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture
def server() -> Iterator[str]:
    _Handler.statuses = []
    _Handler.seen = []
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def test_request_follows_chat_completions_protocol(server: str) -> None:
    provider = OpenAICompatibleProvider(server + "/", "local-model", api_key="k-123")
    answer = provider.complete([{"role": "user", "content": "ping"}], max_tokens=7)
    assert answer == "pong"
    [call] = _Handler.seen
    assert call["path"] == "/v1/chat/completions"
    assert call["auth"] == "Bearer k-123"
    assert call["body"] == {
        "model": "local-model",
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.0,
        "max_tokens": 7,
    }


def test_retries_server_errors_then_succeeds(server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("faithgate.llm.time.sleep", lambda s: None)
    _Handler.statuses = [503, 429]
    provider = OpenAICompatibleProvider(server, "m", api_key="")
    assert provider.complete([{"role": "user", "content": "x"}]) == "pong"
    assert len(_Handler.seen) == 3
    assert _Handler.seen[0]["auth"] is None


def test_client_errors_are_not_retried(server: str) -> None:
    _Handler.statuses = [400]
    with pytest.raises(ProviderError, match="HTTP 400"):
        OpenAICompatibleProvider(server, "m", api_key="").complete(
            [{"role": "user", "content": "x"}]
        )
    assert len(_Handler.seen) == 1
