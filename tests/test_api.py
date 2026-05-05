from fastapi.testclient import TestClient

from src.churn.api import app

client = TestClient(app)


def get_sample_customer() -> dict:
    """Retorna um exemplo válido de cliente para teste da API."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 845.5,
    }


def test_health_endpoint() -> None:
    """Verifica se o endpoint /health responde corretamente."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True
    assert response.json()["model_name"] == "pytorch_mlp"


def test_predict_endpoint() -> None:
    """Verifica se o endpoint /predict retorna uma predição válida."""
    response = client.post("/predict", json=get_sample_customer())

    assert response.status_code == 200

    data = response.json()

    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "model_name" in data
    assert "model_version" in data

    assert data["churn_prediction"] in [0, 1]
    assert 0 <= data["churn_probability"] <= 1
    assert data["risk_level"] in ["baixo", "medio", "alto"]
    assert data["model_name"] == "pytorch_mlp"