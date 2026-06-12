import json
import mlflow
import logging
import os
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

# Set up MLflow tracking URI
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://ec2-3-89-143-153.compute-1.amazonaws.com:5000/")
mlflow.set_tracking_uri(tracking_uri)

# Logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

def load_model_info(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)

def register_model(model_name: str, model_info: dict):
    """Register the model using a direct S3 path to bypass proxy issues."""
    try:
        # Use your verified path directly.
        # Format: s3://{bucket}/{experiment_id}/{run_id}/artifacts/{model_name}
        s3_bucket = os.getenv("S3_BUCKET_NAME", "mlflow-bucket-4545")
        experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", "1")
        run_id = model_info['run_id']
        model_path = model_info['model_path']
        
        # Determine if we should use direct S3 or the provided path
        # If the path is already a full URI, use it. Otherwise, construct it.
        if model_path.startswith("s3://") or model_path.startswith("mlflow-artifacts:/"):
            model_uri = model_path
        else:
            model_uri = f"s3://{s3_bucket}/{experiment_id}/{run_id}/artifacts/lgbm_model"
        
        logger.debug(f'Registering model from URI: {model_uri}')
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition stage
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        logger.info(f'Success: Model {model_name} version {model_version.version} registered.')
    except Exception as e:
        logger.error(f'Registration failed: {e}')
        raise

def main():
    try:
        model_info = load_model_info('experiment_info.json')
        register_model("my_model", model_info)
    except Exception as e:
        print(f"Process failed: {e}")

if __name__ == '__main__':
    main()