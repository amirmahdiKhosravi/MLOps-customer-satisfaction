import logging
from zenml import step
import pandas as pd
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow
from zenml.client import Client

Client().activate_stack("mlflow")
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame, 
    config: ModelNameConfig
) -> RegressorMixin:
    """
    Training the model on the given dataframe.

    Args: 
        df: the given dataframe for training.
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error(f"Error in train_model: {e}")
        raise e