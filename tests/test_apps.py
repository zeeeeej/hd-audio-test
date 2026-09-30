import asyncio
import json
from contextlib import asynccontextmanager

import httpx
from fastapi.testclient import TestClient

import src.business.main as business
import src.tts_worker.main as tts
import src.voice_gateway.main as gateway


class FakeAsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


class FakeStreamingClient:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    @asynccontextmanager
    async def stream(self, method: str, url: str, **kwargs):
        yield self.response


def test_business_page_and_voice_list() -> None:
    client = TestClient(business.app)
    page = client.get("/")
    assert page.status_code == 200
    assert "Voice 服务测试" in page.text
    voices = client.get("/api/voices").json()
    assert voices["default"] == "Vivian"
    assert "Uncle_Fu" in voices["voices"]


def test_business_rejects_unknown_voice_before_proxy() -> None:
    client = TestClient(business.app)
    response = client.post("/api/speech", json={"input": "你好", "voice": "unknown"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_voice"

    streamed = client.post("/api/speech/stream", json={"input": "你好", "voice": "unknown"})
    assert streamed.status_code == 422
    assert streamed.json()["error"]["code"] == "invalid_voice"


def test_business_rejects_invalid_hotwords() -> None:
    client = TestClient(business.app)
    response = client.post(
        "/api/transcriptions",
        data={"language": "Chinese", "hotwords": "not-json"},
        files={"file": ("sample.wav", b"not-audio", "audio/wav")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_hotwords"


def test_gateway_requires_bearer_token(monkeypatch) -> None:
    monkeypatch.setattr(gateway, "VOICE_API_TOKEN", "secret")
    client = TestClient(gateway.app)
    denied = client.post("/v1/audio/speech", json={"input": "你好"})
    assert denied.status_code == 401
    assert denied.json()["error"]["code"] == "unauthorized"
    assert gateway._authorized("Bearer secret")
    assert not gateway._authorized("Bearer wrong")

    streamed = client.post("/v1/audio/speech/stream", json={"input": "你好"})
    assert streamed.status_code == 401
    assert streamed.json()["error"]["code"] == "unauthorized"


def test_default_hotwords_merge_without_duplicates(monkeypatch) -> None:
    monkeypatch.setattr(business, "DEFAULT_HOTWORDS_RAW", json.dumps(["东方树叶", "冰露"]))
    assert business._combined_hotwords(json.dumps(["冰露", "元气森林"])) == [
        "东方树叶",
        "冰露",
        "冰露饮用水",
        "东方树叶茉莉花茶",
        "元气森林白桃味气泡水",
        "可口可乐",
        "元气森林",
    ]


def test_tts_stream_events_are_ordered(monkeypatch) -> None:
    calls: list[str] = []

    def synthesize(text: str, voice: str, language: str, instruct: str) -> tuple[bytes, int]:
        calls.append(text)
        return f"wav:{text}".encode(), 24000

    monkeypatch.setattr(tts.runtime, "synthesize_text", synthesize)
    payload = tts.SpeechRequest(input="第一句。第二句！")

    async def collect() -> list[dict[str, object]]:
        iterator = tts._speech_events(payload, "request-1")
        start = json.loads(await anext(iterator))
        assert calls == []
        first_audio = json.loads(await anext(iterator))
        assert calls == ["第一句。"]
        remaining = [json.loads(line) async for line in iterator]
        return [start, first_audio, *remaining]

    events = asyncio.run(collect())
    assert calls == ["第一句。", "第二句！"]
    assert [event["type"] for event in events] == ["start", "audio", "audio", "done"]
    assert events[0]["request_id"] == "request-1"
    assert events[0]["sentence_count"] == 2
    assert events[1]["index"] == 0
    assert events[1]["sample_rate"] == 24000
    assert events[3]["sentence_count"] == 2


def test_tts_stream_reports_failure_after_started_audio(monkeypatch) -> None:
    def synthesize(text: str, voice: str, language: str, instruct: str) -> tuple[bytes, int]:
        if text.startswith("第二"):
            raise RuntimeError("test failure")
        return b"wav", 24000

    monkeypatch.setattr(tts.runtime, "synthesize_text", synthesize)
    payload = tts.SpeechRequest(input="第一句。第二句！")

    async def collect() -> list[dict[str, object]]:
        return [json.loads(line) async for line in tts._speech_events(payload, "request-2")]

    events = asyncio.run(collect())
    assert [event["type"] for event in events] == ["start", "audio", "error"]
    assert events[-1]["index"] == 1
    assert events[-1]["error"]["code"] == "tts_failed"


def test_gateway_forwards_stream_without_changing_event_body(monkeypatch) -> None:
    chunks = [b'{"type":"start","sentence_count":1}\n', b'{"type":"done"}\n']
    upstream = httpx.Response(
        200,
        headers={"Content-Type": "application/x-ndjson", "X-Request-ID": "upstream-id"},
        stream=FakeAsyncStream(chunks),
    )
    monkeypatch.setattr(gateway, "VOICE_API_TOKEN", "secret")
    monkeypatch.setattr(
        gateway.httpx,
        "AsyncClient",
        lambda **kwargs: FakeStreamingClient(upstream),
    )

    response = TestClient(gateway.app).post(
        "/v1/audio/speech/stream",
        headers={"Authorization": "Bearer secret"},
        json={"input": "你好。"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "upstream-id"
    assert response.content == b"".join(chunks)


def test_business_forwards_stream_without_changing_event_body(monkeypatch) -> None:
    chunks = [b'{"type":"start","sentence_count":1}\n', b'{"type":"done"}\n']
    upstream = httpx.Response(
        200,
        headers={"Content-Type": "application/x-ndjson", "X-Request-ID": "gateway-id"},
        stream=FakeAsyncStream(chunks),
    )
    monkeypatch.setattr(
        business.httpx,
        "AsyncClient",
        lambda **kwargs: FakeStreamingClient(upstream),
    )

    response = TestClient(business.app).post(
        "/api/speech/stream",
        json={"input": "你好。", "voice": "Vivian"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "gateway-id"
    assert response.content == b"".join(chunks)
