import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load iris dataset
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model params
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Test the model
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

# Init mlflow tracking
mlflow.set_experiment("iris-classification")
mlflow.set_tracking_uri("http://localhost:5000")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    # Set tag
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the signature of the model
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr, 
        input_example=X_train,
        signature=signature,
        registered_model_name="tracking_quickstart",
        artifact_path="iris_model"
    )

# Load the model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

# Predict on the test set
y_pred = loaded_model.predict(X_test)

# Get feature names
feature_names = datasets.load_iris().feature_names

# Result
result = pd.DataFrame(X_test, columns=feature_names)
result["actual_class"] = y_test
result["predicted_class"] = y_pred

# Print the result
result[:4]