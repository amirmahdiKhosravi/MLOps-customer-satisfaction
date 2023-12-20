import logging
from zenml import step
import pandas as pd
from src.evaluation import MSE, R2, RMSE
import numpy as np
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated

@step
def evaluate_model(model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[
    Annotated[float, "r2"],
    Annotated[float, "rmse"]
]:
    """
    
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        return r2, rmse
    except Exception as e:
        logging.error(f"Error in evaluate_model: {e}")
        raise e



