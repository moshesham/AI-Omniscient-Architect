import pytest
from fastapi.testclient import TestClient
from omniscient_api.app import create_app
from omniscient_api.config import get_config

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

def test_auth_disabled_by_default(client):
    # Should succeed without key (422 because body missing, but not 403/401)
    response = client.post("/api/v1/analyze", json={})
    assert response.status_code != 403
    assert response.status_code != 401

def test_auth_enabled_missing_key(monkeypatch):
    monkeypatch.setenv("OMNISCIENT_API_KEY", "secret-key")
    # Clear cache to reload config
    get_config.cache_clear()
    
    app = create_app()
    client = TestClient(app)
    
    response = client.post("/api/v1/analyze", json={})
    assert response.status_code == 401

def test_auth_enabled_invalid_key(monkeypatch):
    monkeypatch.setenv("OMNISCIENT_API_KEY", "secret-key")
    get_config.cache_clear()
    
    app = create_app()
    client = TestClient(app)
    
    response = client.post("/api/v1/analyze", json={}, headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 403

def test_auth_enabled_valid_key(monkeypatch):
    monkeypatch.setenv("OMNISCIENT_API_KEY", "secret-key")
    get_config.cache_clear()
    
    app = create_app()
    client = TestClient(app)
    
    # 422 Unprocessable Entity means it passed auth but failed validation
    response = client.post("/api/v1/analyze", json={}, headers={"X-API-Key": "secret-key"})
    assert response.status_code == 422
