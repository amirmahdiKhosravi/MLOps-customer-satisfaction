import logging
from zenml import step
import pandas as pd

@step
def clean_df(df: pd.DataFrame) -> None:
    """
    cleaning data from given dataframe.

    Args:
        df:
            the dataframe to clean
    Returns:
        pd.DataFrame: cleaned dataframe.
    """
    pass