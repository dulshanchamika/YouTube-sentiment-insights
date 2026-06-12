import json
import mlflow
import logging
from mlflow.tracking import MlflowClient

# Set up MLflow tracking URI
mlflow.set_tracking_uri("http://ec2-3-89-143-153.compute-1.amazonaws.com:5000/")

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
        s3_bucket = "mlflow-bucket-4545"
        experiment_id = "1"
        run_id = model_info['run_id']
        model_path = model_info['model_path']
        
        # Hard-coded direct S3 URI
        model_uri = f"s3://{s3_bucket}/{experiment_id}/{run_id}/artifacts/{model_path}"
        
        logger.debug(f'Registering model directly from S3: {model_uri}')
        
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