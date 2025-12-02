# MLFlow Quickstart

[Demo video](https://www.youtube.com/watch?v=cjeCAoW83_U) *not very good*

[Docs Quickstart](https://mlflow.org/docs/latest/ml/tracking/quickstart/)

## 1. Create venv

```bash
python -m venv venv
source venv/bin/activate
```

Install packages
```bash
pip install -r requirements.txt
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
import mlflow
mlflow.autolog()
```

Best practice: be more specific about the model library you're using with:
```python
mlflow.sklearn.autolog()
```
You can view the supported libraries [here](https://mlflow.org/docs/latest/ml/tracking/autolog/#supported-libraries)

## 2. Start MLFlow server

```bash
mlflow ui --port 5000
```