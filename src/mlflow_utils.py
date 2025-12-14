import mlflow
import os

def setup_mlflow():
    """
    Configures MLflow tracking URI.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI set to: {tracking_uri}")

def log_experiment_params(params):
    """
    Logs a dictionary of parameters to MLflow.
    """
    mlflow.log_params(params)

def log_experiment_metrics(metrics):
    """
    Logs a dictionary of metrics to MLflow.
    """
    mlflow.log_metrics(metrics)
