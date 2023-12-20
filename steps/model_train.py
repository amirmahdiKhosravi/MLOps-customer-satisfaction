import logging
from zenml import step
import pandas as pd
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
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
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error(f"Error in train_model: {e}")
        raise e