import logging
from zenml import step
import pandas as pd

@step
def train_model(df: pd.DataFrame) -> None:
    """
    Training the model on the given dataframe.

    Args: 
        df: the given dataframe for training.
    """
    pass