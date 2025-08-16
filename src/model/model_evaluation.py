import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models import infer_signature
import json

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise

def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise

def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and return metrics and confusion matrix."""
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.debug('Model evaluation completed')
        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def main():
    mlflow.set_tracking_uri("http://ec2-13-51-196-160.eu-north-1.compute.amazonaws.com:5000/")
    mlflow.set_experiment('dvc-pipeline-runs')

    with mlflow.start_run() as run:
        try:
            # Load parameters
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Load model + vectorizer
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Load test data
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            # Prepare test data
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Input example + signature
            input_example = pd.DataFrame(
                X_test_tfidf.toarray()[:5],
                columns=vectorizer.get_feature_names_out()
            )
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))

            # Evaluate model
            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            # Log metrics
            mlflow.log_metric("accuracy", report["accuracy"])
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metric(f"precision_{label}", metrics["precision"])
                    mlflow.log_metric(f"recall_{label}", metrics["recall"])
                    mlflow.log_metric(f"f1_{label}", metrics["f1-score"])

            # Log confusion matrix
            cm_file_path = "confusion_matrix_TestData.png"
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - Test Data')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(cm_file_path)
            plt.close()
            mlflow.log_artifact(cm_file_path)

            # Log vectorizer
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Log model - FIXED: Replaced 'name' with 'artifact_path'
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="lgbm_model",  # This creates the folder structure
                signature=signature,
                input_example=input_example
            )

            # Add tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")
            mlflow.set_tag("mlflow_version", mlflow.__version__)

            # Save run info for DVC compatibility
            model_info = {
                "run_id": run.info.run_id,
                "artifact_uri": mlflow.get_artifact_uri(),
                "experiment_id": run.info.experiment_id,
                "run_name": run.info.run_name,
                "lifecycle_stage": run.info.lifecycle_stage,
                "model_path": f"runs:/{run.info.run_id}/lgbm_model",
                "tracking_uri": mlflow.get_tracking_uri(),
                "experiment_name": "dvc-pipeline-runs"
            }

            # Save to JSON file for DVC compatibility
            experiment_info_path = os.path.join(root_dir, "experiment_info.json")
            with open(experiment_info_path, "w") as f:
                json.dump(model_info, f, indent=4)

            logger.info(f"Experiment info saved to {experiment_info_path}")
            logger.info(f"Run ID: {run.info.run_id}")
            logger.info(f"Model URI: runs:/{run.info.run_id}/lgbm_model")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            raise

if __name__ == '__main__':
    main()