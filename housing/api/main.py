import os
import logging
import sqlite3
import json
from datetime import datetime

from flask import Flask, request, jsonify, g
import pandas as pd
import mlflow

# Configure MLflow tracking
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://192.168.0.206:5000")
)

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLite DB setup
DATABASE = "logs.db"


def get_db():
    """Get a SQLite database connection and initialize tables if needed."""
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_data TEXT NOT NULL,
                predictions TEXT NOT NULL
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                count INTEGER NOT NULL
            )
            """
        )
        cur = db.execute(
            "SELECT count(*) FROM metrics WHERE endpoint = '/predict'"
        )
        if cur.fetchone()[0] == 0:
            db.execute(
                "INSERT INTO metrics (endpoint, count) VALUES (?, ?)",
                ('/predict', 0),
            )
        db.commit()
    return db


@app.teardown_appcontext
def close_connection(exception):
    """Close the database connection on app context teardown."""
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def increment_metric(endpoint):
    """Increment request count for an endpoint."""
    db = get_db()
    db.execute(
        "UPDATE metrics SET count = count + 1 WHERE endpoint = ?",
        (endpoint,),
    )
    db.commit()


# Load model once
model_name = "best_housing_model"
logger.info(f"Loading model: {model_name}")
model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict endpoint for housing prices."""
    if not request.is_json:
        logger.error("Request content-type is not application/json")
        return jsonify(
            {"error": "Request content-type must be application/json"}
        ), 400

    data = request.get_json()
    if not data:
        logger.error("No input data provided")
        return jsonify({"error": "No input data provided"}), 400

    try:
        # Wrap single record dictionary in a list if needed
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        else:
            input_df = pd.DataFrame(data)

        expected_columns = [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude",
        ]
        input_df = input_df.astype(
            {col: "float64" for col in expected_columns}
        )

        logger.info(
            "Received input for prediction: %s",
            input_df.to_dict(orient='records'),
        )
        preds = model.predict(input_df)
        logger.info("Model predictions: %s", preds.tolist())

        timestamp = datetime.utcnow().isoformat()
        input_json = json.dumps(data)
        preds_json = json.dumps(preds.tolist())
        db = get_db()
        db.execute(
            """
            INSERT INTO prediction_logs
            (timestamp, input_data, predictions)
            VALUES (?, ?, ?)
            """,
            (timestamp, input_json, preds_json),
        )
        db.commit()

        increment_metric('/predict')

        return jsonify({"predictions": preds.tolist()})
    except Exception as e:
        logger.error("Prediction error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return endpoint usage metrics."""
    db = get_db()
    cur = db.execute("SELECT endpoint, count FROM metrics")
    metrics_data = {row[0]: row[1] for row in cur.fetchall()}
    return jsonify(metrics_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
