from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_portfolio_assessment_endpoint():
    payload = {
        "portfolio_id": "DEMO-001",
        "cash": 2000,
        "holdings": [
            {"symbol": "ALFA", "quantity": 10, "price": 100},
            {"symbol": "BETA", "quantity": 5, "price": 200},
        ],
    }

    with TestClient(app) as client:
        response = client.post("/portfolio/assess", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert result["total_value"] == 4000
    assert result["passed"] is True
    assert result["violations"] == []


def test_invalid_holding_is_rejected():
    payload = {
        "portfolio_id": "DEMO-001",
        "cash": 2000,
        "holdings": [
            {"symbol": "ALFA", "quantity": -1, "price": 100}
        ],
    }

    with TestClient(app) as client:
        response = client.post("/portfolio/assess", json=payload)

    assert response.status_code == 422
