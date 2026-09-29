import json

from fastapi.testclient import TestClient

import src.business.main as business
import src.voice_gateway.main as gateway


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


def test_default_hotwords_merge_without_duplicates(monkeypatch) -> None:
    monkeypatch.setattr(business, "DEFAULT_HOTWORDS_RAW", json.dumps(["东方树叶", "冰露"]))
    assert business._combined_hotwords(json.dumps(["冰露", "元气森林"])) == [
        "东方树叶",
        "冰露",
        "元气森林",
    ]
