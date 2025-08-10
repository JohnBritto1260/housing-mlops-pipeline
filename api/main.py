from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("models/DecisionTree.pkl")


class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.post("/predict")
def predict(data: HouseFeatures):
    """Predict house value from features."""
    features = np.array([[v for v in data.dict().values()]])
    prediction = model.predict(features)[0]
    return {"prediction": float(prediction)}
