import os
import logging
from logging.handlers import RotatingFileHandler
import sqlite3
import json
from datetime import datetime

from flask import Flask, request, jsonify, g, Response
import pandas as pd
import mlflow
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# ------------------------------
# 1. Setup Logging
# ------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file_path = os.path.join(LOG_DIR, "app.log")
file_handler = RotatingFileHandler(
    log_file_path, maxBytes=5 * 1024 * 1024, backupCount=5
)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO, handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)

# ------------------------------
# 2. MLflow Model Setup
# ------------------------------
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://192.168.0.206:5000")
)
model_name = "best_housing_model"
logger.info(f"Loading MLflow model: {model_name}")
model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")

# ------------------------------
# 3. Flask App & Database Setup
# ------------------------------
app = Flask(__name__)
DATABASE = "logs.db"

# Prometheus Counter
PREDICT_REQUESTS = Counter(
    "housing_predict_requests_total", "Total number of /predict requests"
)


def get_db():
    """Get SQLite connection and initialize tables."""
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
        db.commit()
    return db


@app.teardown_appcontext
def close_connection(exception):
    """Close database connection."""
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


# ------------------------------
# 4. Routes
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """Make predictions and log them."""
    if not request.is_json:
        logger.error("Invalid content-type: must be JSON")
        return (
            jsonify(
                {"error": "Request content-type must be application/json"}
            ),
            400,
        )

    data = request.get_json()
    if not data:
        logger.error("Empty request body")
        return jsonify({"error": "No input data provided"}), 400

    try:
        # Convert input to DataFrame
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        else:
            input_df = pd.DataFrame(data)

        expected_columns = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ]
        input_df = input_df.astype(
            {col: "float64" for col in expected_columns}
        )

        # Log input
        logger.info(f"Prediction input: {input_df.to_dict(orient='records')}")
        preds = model.predict(input_df)
        logger.info(f"Prediction output: {preds.tolist()}")

        # Save to SQLite
        timestamp = datetime.utcnow().isoformat()
        db = get_db()
        db.execute(
            "INSERT INTO prediction_logs (timestamp, input_data, predictions) VALUES (?, ?, ?)",
            (timestamp, json.dumps(data), json.dumps(preds.tolist())),
        )
        db.commit()

        # Increment Prometheus metric
        PREDICT_REQUESTS.inc()

        return jsonify({"predictions": preds.tolist()})
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


# ------------------------------
# 5. Main Entry
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
