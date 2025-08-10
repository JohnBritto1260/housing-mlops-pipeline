from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Dummy prediction logic
    prediction_value = (
        data["MedInc"] * 0.5
        + data["HouseAge"] * 0.3
        + data["AveRooms"] * 0.2
    )

    return jsonify({"prediction": float(prediction_value)})


if __name__ == "__main__":
    app.run(debug=True)
