import json

import httpx
import pytest
from fastapi.testclient import TestClient

from privacy_bench.serve.app import WARMUP_QUERY, build_user_content, create_app

SETTINGS = {
    "backends": {"base": "http://base", "lora": "http://lora"},
    "arms": {
        "A0": {"backend": "base", "retrieval": False},
        "A1": {"backend": "lora", "retrieval": False},
        "A2": {"backend": "base", "retrieval": True},
    },
    "retrieval": {"method": "hybrid", "k": 2},
    "generation": {"max_tokens": 16, "timeout_seconds": 5},
    "system_prompt": "SYS",
}
CHUNKS = [
    {"chunk_id": "s:6#c000", "uid": "s:6", "display": "s 6", "title": "Interpretation", "text": "Definitions."},
    {"chunk_id": "s:7#c000", "uid": "s:7", "display": "s 7", "title": "Acts and practices", "text": "Scope."},
]


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, method, k):
        self.calls.append((query, method, k))
        return {"chunks": CHUNKS[:k], "retrieval_ms": 1.5}


def make_client(seen):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        body = json.loads(request.content)
        seen.append((request.url.host, body))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": " Answer. [Source: s 6] "}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 6},
            "timings": {"prompt_ms": 12.0, "predicted_ms": 80.0},
        })

    retriever = FakeRetriever()
    app = create_app(SETTINGS, retriever=retriever, transport=httpx.MockTransport(handler))
    return TestClient(app), retriever


def test_context_layout():
    text = build_user_content("What is X?", CHUNKS)
    assert text.startswith("Use the following extracts")
    assert "[1] s 6 Interpretation\nDefinitions." in text and text.endswith("Question: What is X?")
    assert build_user_content("Q?", []) == "Q?"


def test_closed_book_arm_routes_to_backend_without_retrieval():
    seen = []
    client, retriever = make_client(seen)
    with client:
        r = client.post("/answer", json={"question": "Q?", "arm": "A1"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "Answer. [Source: s 6]" and data["retrieved"] == []
    assert seen[0][0] == "lora" and seen[0][1]["messages"][1]["content"] == "Q?"
    assert seen[0][1]["temperature"] == 0 and retriever.calls == [(WARMUP_QUERY, "hybrid", 2)]
    assert data["latency"]["prefill_ms"] == 12.0 and data["tokens"]["completion"] == 6


def test_rag_arm_adds_context_and_reports_retrieval():
    seen = []
    client, retriever = make_client(seen)
    with client:
        r = client.post("/answer", json={"question": "Q?", "arm": "A2"})
    data = r.json()
    assert retriever.calls == [(WARMUP_QUERY, "hybrid", 2), ("Q?", "hybrid", 2)]
    assert [c["uid"] for c in data["retrieved"]] == ["s:6", "s:7"]
    assert seen[0][0] == "base" and "[2] s 7 Acts and practices" in seen[0][1]["messages"][1]["content"]
    assert data["latency"]["retrieval_ms"] == 1.5


@pytest.mark.parametrize("payload,status", [({"question": "Q?", "arm": "A9"}, 422), ({"question": "", "arm": "A0"}, 422)])
def test_rejects_bad_requests(payload, status):
    client, _ = make_client([])
    with client:
        assert client.post("/answer", json=payload).status_code == status


def test_health_reports_backends():
    client, _ = make_client([])
    with client:
        assert client.get("/health").json() == {"ok": True, "backends": {"base": True, "lora": True}}
