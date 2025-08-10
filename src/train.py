import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlflow.models.signature import infer_signature

# Set the MLflow tracking URI (your local MLflow server)
mlflow.set_tracking_uri("http://localhost:5000")


def load_data():
    """Load California housing dataset."""
    return pd.read_csv("data/raw/california.csv")


def train_and_register_model(model, model_name, X_train, y_train, X_test, y_test):
    """Train, log, and register a model with MLflow."""
    with mlflow.start_run(run_name=model_name) as run:
        print(f"🏃 Training and logging model: {model_name}")

        # Train
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        rmse = mean_squared_error(y_test, preds) ** 0.5
        mae = mean_absolute_error(y_test, preds)

        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        # Log model
        artifact_path = f"{model_name}_model"
        input_example = X_test[:1]
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            input_example=input_example,
            signature=signature,
        )

        print(f"✅ Logged model: {artifact_path}")

        # Register model in MLflow Model Registry
        result = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/{artifact_path}",
            name="best_housing_model",
        )

        print(
            f"✅ Successfully registered model: {result.name} "
            f"(version {result.version})"
        )
        print(
            f"📦 View run {model_name} at: "
            f"{mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
        )


if __name__ == "__main__":
    # Load and split data
    df = load_data()
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train and register both models
    train_and_register_model(
        LinearRegression(), "LinearRegression", X_train, y_train, X_test, y_test
    )
    train_and_register_model(
        DecisionTreeRegressor(max_depth=5),
        "DecisionTree",
        X_train,
        y_train,
        X_test,
        y_test,
    )
