import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingestng data from data path.
    """
    def __init__(self, data_path: str):
        """
        instanciating the data path.

        Args:
            data_path: the path to the data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Getting data from the given path.

        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data path.

    Args:
        data_path: path to the data
    Returns: 
        pd.DataFrame: the ingested data
    """

    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while Ingesting data: {e}")
        raise e