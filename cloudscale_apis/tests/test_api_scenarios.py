"""
API & Webhook Testing Scenarios
- Covers success, error, and edge cases for endpoints and webhooks.
- Use pytest or similar frameworks for automation.
"""
import pytest
from fastapi.testclient import TestClient
from cloudscale_apis.endpoints.example_api import router as api_router
from cloudscale_apis.webhooks.example_webhook import router as webhook_router
from fastapi import FastAPI

@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    app.include_router(webhook_router, prefix="/hooks")
    return TestClient(app)

def test_evaluate_success(client):
    resp = client.post("/api/evaluate", json={"text": "test"})
    assert resp.status_code == 200
    assert "result" in resp.json()

def test_evaluate_error(client):
    resp = client.post("/api/evaluate", json={})
    assert resp.status_code == 400

def test_webhook_receive(client):
    resp = client.post("/hooks/webhook/event", json={"event": "test"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "received"
