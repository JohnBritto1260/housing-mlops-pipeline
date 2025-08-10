import sys
import os
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "MedInc": 8.5,
        "HouseAge": 20,
        "AveRooms": 5.2,
        "AveBedrms": 1.1,
        "Population": 850,
        "AveOccup": 3.1,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }

    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)
