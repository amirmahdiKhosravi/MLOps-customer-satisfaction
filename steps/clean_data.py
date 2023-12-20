import logging
from zenml import step
import pandas as pd
from src.data_cleaning import DataCleaing, DataDivideStrategy, DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"], 
    Annotated[pd.Series, "y_train"], 
    Annotated[pd.Series, "y_test"]]:

    """
    Cleans data and splits it into train and test sets

    Args:
        df: raw data

    Returns:
        X_train: train data
        X_test: test data
        y_train: train targets
        y_test: test targets
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaing(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaing(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in cleaning data in clean data step: {e}")
        raise e