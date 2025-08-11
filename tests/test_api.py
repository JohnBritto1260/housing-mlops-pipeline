import pytest
from housing.api.main import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_predict_success(client):
    """Test prediction endpoint with valid data."""
    payload = [{
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.02381,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }]
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)


def test_predict_no_data(client):
    """Test prediction endpoint with no data."""
    response = client.post("/predict", json=None)
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_metrics_endpoint(client):
    """Test metrics endpoint returns correct format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.data.decode("utf-8")
    assert "housing_predict_requests_total" in data
    assert "predict" in data
