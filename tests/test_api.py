import sys
import os
import json
import pytest
from housing.api.main import app

# Ensure project root is in sys.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


@pytest.fixture
def client():
    """Flask test client fixture."""
    with app.test_client() as client:
        yield client


def test_predict_endpoint(client):
    """Test the /predict endpoint."""
    payload = {
        "MedInc": 8.4,
        "HouseAge": 20,
        "AveRooms": 5.2,
        "AveBedrms": 1.1,
        "Population": 850,
        "AveOccup": 3.1,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }

    res = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )

    assert res.status_code == 200
    body = res.get_json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)
