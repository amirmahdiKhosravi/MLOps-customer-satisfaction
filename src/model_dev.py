from abc import ABC, abstractmethod
import logging
from sklearn.linear_model import LinearRegression

class Model(ABC):

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model

        Args:
            X_train: data to train model
            y_train: target values to train model

        Returns:
            None
        """
        pass

class LinearRegressionModel(Model):

    def train(self, X_train, y_train, **kwargs):
        """
        Train the model

        Args:
            X_train: data to train model
            y_train: target values to train model

        Returns:
            None
        """
        try:

            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed.")
            return reg
        except Exception as e:
            logging.error(f"Error in train function: {e}")
            raise e