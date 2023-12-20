import logging
from abc import ABC, abstractclassmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class to define strategy to evaluate our model
    """
    @abstractclassmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        calculating scores for the model

        Args: 
            y_true: the actual target values 
            y_pred: the predicted target values
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    this is evaluation strategy which do the mean square error.
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e

class R2(Evaluation):

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("calculating R2")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in R2 calculation: {e}")
            raise e

class RMSE(Evaluation):
    """
    this is evaluation strategy which do the mean square error.
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("calculating MSE")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info(f"MSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e

