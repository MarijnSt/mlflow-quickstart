# MLFlow Quickstart

[Demo video](https://www.youtube.com/watch?v=cjeCAoW83_U) *not very good*

[Docs Quickstart](https://mlflow.org/docs/latest/ml/tracking/quickstart/)

## 1. Set up ML flow
Create a venv for the project:
```bash
python -m venv venv
source venv/bin/activate
```

Install packages:
```bash
pip install -r requirements.txt
```

Create a file where the model will be made and give the experiment a name:
```python
import mlflow

mlflow.set_experiment("MLflow Quickstart")
```

## 2. Prepare training data

In this example we'll be using the Iris dataset.

## 3. Train the model and track with autolog

One feature of ML flow is autolog which allows you to focus on building and training the model and let ML Flow handle the rest:
* **Saving** the trained model.
* Recording the model's **performance metrics** during training, such as accuracy, precision, AUC curve.
* Logging **hyperparameter values** used to train the model.
* Track **metadata** such as input data format, user, timestamp, etc.
More information about autologging can be found [here](https://mlflow.org/docs/latest/ml/tracking/autolog/)

To use autolog, just add this code before training the model:
```python
mlflow.autolog()
```

Best practice: be more specific about the model library you're using with:
```python
mlflow.sklearn.autolog()
```
You can view the supported libraries [here](https://mlflow.org/docs/latest/ml/tracking/autolog/#supported-libraries)

## 4. Start MLFlow server

To see the results of training, you can start the server:
```bash
mlflow ui --port 5000
```

In this UI you can see the new experiment and when you click on it, you can see the different runs.
On the run page you can see all the logged data and you can find the model which also has a page with more information.


## 5. Manually log data

Besides the autolog, we can also set up runs and tracking of hyperparameters, model and metrics ourselves.
Tag the run to easily find it in the UI:

```python
# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Log the model
    model_info = mlflow.sklearn.log_model(sk_model=lr, name="iris_model")

    # Predict on the test set, compute and log the loss metric
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Optional: Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")
```

## 6. Get inference

Once the model is trained, we can get inference from it with new data.
Wether we're using the autolog or our own custom runs doesn't really matter. This new run will also be logged.