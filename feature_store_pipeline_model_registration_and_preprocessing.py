import mlflow
import logging
import sys
import os

sys.path.append(os.path.abspath('../'))
from reference import *
print(ENV_VARS, MODELS_NAME)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_task_values():
    try:
        run_id = dbutils.jobs.taskValues.get(taskKey="model_retraining_task_2", key="run_id")
        print(f"Received run_id: {run_id}")

        model_uri = dbutils.jobs.taskValues.get(taskKey="model_retraining_task_2", key="model_uri")
        print(f"Received model_uri: {model_uri}")

        return run_id, model_uri
    except Exception as e:
        logger.error(f"Error getting task values: {e}")
        raise


def register_and_log_model(run_id: str, model_uri: str, model_name: str):
    try:
        with mlflow.start_run(run_id=run_id) as run:
            # Set MLflow Registry URL
            mlflow.set_registry_uri("databricks-uc")
            # Register the model        
            registered_model = mlflow.register_model(
                model_uri,
                f"{ENV_VARS['CATALOG']}.{ENV_VARS['SCHEMA']}.{model_name}"
            )
            
            print(f"Model '{registered_model.name}' registered successfully with version '{registered_model.version}' with URI: {model_uri} in Unity catalog")

            # Return the model name and version
            return model_name, registered_model.version

    except mlflow.exceptions.MlflowException as e:
        if "Model with name" in str(e):
            print(f"Model '{model_name}' already exists. Consider using a different name or version.")
        else:
            logger.error(f"Error registering model: {e}")
            raise e
    except Exception as e:
        logger.error(f"Unexpected error during model registration: {e}")
        raise e


def main():
    try:
        # run_id, model_name, model_uri = get_task_values()
        run_id, model_uri = get_task_values()
        print(f"Received run_id: {run_id}, model_name: {MODELS_NAME['MODEL2']}, model_uri: {model_uri}")

        # Register and log the model
        registered_model_name, registered_model_version = register_and_log_model(
            run_id,
            model_uri,
            MODELS_NAME['MODEL2']
        )
        dbutils.jobs.taskValues.set("registered_model_name_2", registered_model_name)
        dbutils.jobs.taskValues.set("registered_model_version_2", registered_model_version)
        print(f"Model '{registered_model_name}' registered successfully with version '{registered_model_version}' with URI: {model_uri}")
    except Exception as e:
        logger.error(f"Error during model registration: {e}")
        raise e
    
if __name__ == "__main__":
    main()
