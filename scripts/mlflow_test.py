import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
import numpy as np

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://ec2-13-51-196-160.eu-north-1.compute.amazonaws.com:5000/")

# Dummy training data
X = np.random.rand(20, 5)
y = np.random.randint(0, 2, 20)

# Train a simple model
model = LogisticRegression()
model.fit(X, y)

# Start an MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("test_param", "log_model_check")

    # Log the sklearn model (⚡ key part)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="dummy_model"
    )

    print("✅ Dummy sklearn model logged to MLflow.")
