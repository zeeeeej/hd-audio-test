import json
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from src.business import main, retail


def payload(**overrides):
    return {"name": "测试饮料", "stock": 2, "price_yuan": "4", **overrides}


def multipart_words(request):
    message = BytesParser(policy=default).parsebytes(
        f'Content-Type: {request.headers["content-type"]}\r\n\r\n'.encode() + request.content
    )
    return next(json.loads(part.get_payload(decode=True)) for part in message.iter_parts()
                if part.get_param("name", header="content-disposition") == "hotwords")


def test_create_edit_persistence_and_legacy_defaults(isolated_catalog):
    client = TestClient(main.app)
    catalog = client.get("/api/retail/catalog").json()
    assert catalog["products"][0]["aliases"] == []
    response = client.post("/api/retail/products", json=payload(aliases=[" 测试 ", "测试"]))
    assert response.status_code == 201
    product = response.json()
    assert product["price_yuan"] == "4.00"
    assert product["aliases"] == ["测试"]
    response = client.patch(f'/api/retail/products/{product["id"]}',
                            json={"stock": 0, "price_yuan": "3.50"})
    assert response.status_code == 200
    # Recreating the provider and HTTP client reads persisted data, no in-memory cache.
    saved = retail.CatalogProvider().load().products[-1]
    assert saved.stock == 0 and saved.price_yuan == "3.50"
    assert saved.name == product["name"] and saved.aliases == ["测试"]
    assert json.loads(isolated_catalog.read_text())["products"][-1] == saved.model_dump()
    assert TestClient(main.app).get("/api/retail/catalog").json()["products"][-1]["stock"] == 0


@pytest.mark.parametrize("values", [
    {"name": " "}, {"name": "字" * 41}, {"stock": -1}, {"stock": 1.5},
    {"stock": True}, {"stock": "2"}, {"price_yuan": "-1"}, {"price_yuan": "1.234"},
    {"price_yuan": True}, {"price_yuan": "NaN"}, {"price_yuan": "1e2"},
    {"aliases": ["字" * 41]}, {"tags": [2]}, {"name": None}, {"id": "replace-id"},
])
def test_invalid_products_do_not_change_file(isolated_catalog, values):
    original = isolated_catalog.read_bytes()
    client = TestClient(main.app)
    assert client.post("/api/retail/products", json=payload(**values)).status_code == 422
    assert client.patch("/api/retail/products/water", json=values).status_code == 422
    assert isolated_catalog.read_bytes() == original


def test_duplicates_missing_and_empty_patch(isolated_catalog):
    client = TestClient(main.app)
    original = isolated_catalog.read_bytes()
    data = payload(name=" 冰露饮用水 ", size="550毫升")
    assert client.post("/api/retail/products", json=data).status_code == 409
    assert client.patch("/api/retail/products/tea", json=data).status_code == 409
    assert client.patch("/api/retail/products/missing", json={"stock": 1}).status_code == 404
    assert client.patch("/api/retail/products/water", json={}).status_code == 422
    assert isolated_catalog.read_bytes() == original


def test_limits_rejected_before_writing(isolated_catalog, monkeypatch):
    client = TestClient(main.app)
    original = isolated_catalog.read_bytes()
    monkeypatch.setenv("DEFAULT_HOTWORDS", json.dumps([f"词{i}" for i in range(98)]))
    response = client.post("/api/retail/products", json=payload())
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "hotwords_config_invalid"
    monkeypatch.setenv("DEFAULT_HOTWORDS", "[]")
    monkeypatch.setenv("DIALOGUE_CONTEXT_CHARS", "501")
    response = client.post("/api/retail/products", json=payload())
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "context_config_invalid"
    assert isolated_catalog.read_bytes() == original


def test_write_failure_preserves_original(isolated_catalog, monkeypatch):
    original = isolated_catalog.read_bytes()

    def fail(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(retail.os, "replace", fail)
    response = TestClient(main.app).post("/api/retail/products", json=payload())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "catalog_save_failed"
    assert isolated_catalog.read_bytes() == original
    assert not list(isolated_catalog.parent.glob(".catalog-*"))


def test_default_catalog_is_seeded_only_once(tmp_path, monkeypatch):
    monkeypatch.delenv("RETAIL_CATALOG_FILE")
    monkeypatch.setattr(retail, "PROJECT_ROOT", tmp_path)
    provider = retail.CatalogProvider()
    provider.load()
    path = tmp_path / "runtime/retail_catalog.json"
    assert path.is_file()
    provider.save_product({"stock": 2, "price_yuan": "4"}, "sparkling")
    loaded = retail.CatalogProvider().load()
    assert loaded.products[2].stock == 2 and loaded.products[2].price_yuan == "4.00"
    path.write_text("broken")
    with pytest.raises(retail.DialogueError, match="配置有误"):
        provider.load()
    assert path.read_text() == "broken"  # Never overwrite corrupt runtime data with seed.


def test_concurrent_additions_are_preserved():
    def add(i):
        return retail.CatalogProvider().save_product(payload(name=f"饮料{i}"))

    with ThreadPoolExecutor(max_workers=4) as executor:
        saved = list(executor.map(add, range(4)))
    catalog = retail.CatalogProvider().load()
    assert {p.id for p in saved} <= {p.id for p in catalog.products}
    assert len(catalog.products) == 8


def test_standalone_asr_sends_latest_catalog_aliases(monkeypatch):
    client = TestClient(main.app)
    response = client.post("/api/retail/products", json=payload(aliases=["测试简称", "测试饮料"]))
    assert response.status_code == 201
    monkeypatch.setattr(main, "DEFAULT_HOTWORDS_RAW", '["测试简称","默认词"]')
    calls = []

    def handler(request):
        calls.append(multipart_words(request))
        return httpx.Response(200, json={"text": "测试饮料"})

    original = httpx.AsyncClient
    monkeypatch.setattr(main.httpx, "AsyncClient", lambda **kw: original(
        transport=httpx.MockTransport(handler), **kw,
    ))

    def transcribe():
        return client.post("/api/transcriptions", data={"hotwords": '["测试简称","手动词"]'},
                           files={"file": ("a.wav", b"audio", "audio/wav")})

    assert transcribe().status_code == 200
    words = calls[-1]
    assert {"测试饮料", "测试简称", "默认词", "手动词"} <= set(words)
    assert len(words) == len(set(words))
    client.patch(f'/api/retail/products/{response.json()["id"]}', json={"aliases": ["新简称"]})
    assert transcribe().status_code == 200
    assert "新简称" in calls[-1]


def test_catalog_failure_is_explicit_in_asr(isolated_catalog):
    Path(isolated_catalog).write_text("invalid")
    response = TestClient(main.app).post("/api/transcriptions", files={
        "file": ("a.wav", b"audio", "audio/wav"),
    })
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "catalog_invalid"
