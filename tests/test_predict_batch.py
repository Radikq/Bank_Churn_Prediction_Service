from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from backend import app

client = TestClient(app)

valid_client = {
    "CustomerId": 15634602,
    "Geography": "France",
    "CreditScore": 619,
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.00,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88,
    "Gender": "Male"
}


def test_predict_batch_success():
    response = client.post("/predict_batch", json={"clients": [valid_client]})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list) and len(data) == 1
    item = data[0]
    for key in {"CustomerId", "Geography", "prediction", "churn_probability"}:
        assert key in item


def test_predict_batch_invalid_geography():
    invalid_client = valid_client.copy()
    invalid_client["Geography"] = "USA"
    response = client.post("/predict_batch", json={"clients": [invalid_client]})
    assert response.status_code == 400


def test_predict_batch_empty_payload():
    response = client.post("/predict_batch", json={"clients": []})
    assert response.status_code == 400
