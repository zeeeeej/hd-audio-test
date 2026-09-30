from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_catalog(tmp_path, monkeypatch):
    """Tests must never read or modify the operator's persisted demo inventory."""
    path = tmp_path / "catalog.json"
    source = Path(__file__).resolve().parents[1] / "src/business/retail_catalog.json"
    path.write_bytes(source.read_bytes())
    monkeypatch.setenv("RETAIL_CATALOG_FILE", str(path))
    return path
