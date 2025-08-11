Assignment 1 – MLOps (S2-24_AIMLCZG523) - Group-97
California Housing (regression) dataset
Overview
This MLOps system implements an end-to-end pipeline for predicting California housing prices. It is fully containerized with Docker, orchestrated via Docker Compose, and monitored using Prometheus. Model versioning and experiment tracking are handled via MLflow, while DVC (Data Version Control) manages dataset versioning and pipeline reproducibility.
GitHub repository link : https://github.com/JohnBritto1260/housing-mlops-pipeline
Docker Hub link : https://hub.docker.com/repository/docker/johnbritto1260/housing-api/general

Architecture Components
    1. API Service (FastAPI)
            1. Exposes /predict endpoint for real-time inference.
            2. Loads the latest registered model from MLflow.
            3. Uses SQLite for request logging.
            4. Exposes /metrics endpoint for Prometheus to scrape operational metrics (e.g., request count, latency).
    2. DVC (Data Version Control)
            1. Manages dataset versioning to ensure reproducibility.
            2. Tracks changes to training data and preprocessing steps.
            3. Integrates with Git for lightweight version tracking without bloating the repository.
            4. Allows rollback to previous dataset versions for consistent experiments.
    3. MLflow Tracking Server
            1. Backend store: SQLite database (mlflow.db).
            2. Artifact store: Local directory (./mlruns).
            3. Tracks experiments, model versions, and training metrics.
    4. Prometheus
            1. Scrapes metrics from the API service at /metrics.
            2. Stores historical metrics for monitoring and alerting.
            3. Configured via prometheus.yml scrape jobs.
ML Pipeline workflow
    1. Data & Model Pipeline
        1. Data Source: California Housing Dataset.
        2. Versioning: Dataset is version-controlled using DVC.
        3. Training: Python scripts train models, logging metrics & artifacts to MLflow.
        4. Model Registry: Best-performing model is promoted to “Production” in MLflow.
        5. Deployment: API loads the latest production model from MLflow at startup.
    2. Containerization & Deployment
        1. Dockerfiles for API service.
        2. docker-compose.yml integrates API, MLflow, and Prometheus.
        3. Images can be published to Docker Hub for portability.
        4. One command (docker compose up) spins up the full stack.
    3. Monitoring & Alerting
    1. Prometheus scrapes /metrics every 15s.
    2. Alerts can be configured for anomalies such as:
                • API error rate > 5%
                • Response latency > 1s
                • Service downtime