from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.predicting_road_accident_risk.exception import CustomException
from src.predicting_road_accident_risk.logger import logger
import os
import sys

@dataclass

# defining where the data will be stored after ingestion
class dataingestionconfig:
    raw_data_path: str=os.path.join("artifacts","raw_data.csv")
    train_data_path: str=os.path.join("artifacts", "train_data.csv")
    test_data_path: str = os.path.join("artifacts", "test_data.csv")

# data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = dataingestionconfig()

    def initiate_data_ingestion(self, df):
        try:
            logger.info("Data ingestion started")

            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path),
                exist_ok=True
            )

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logger.info("Raw data saved successfully")

            return df   #  RETURN ONLY THE DATAFRAME

        except Exception as e:
            raise CustomException(e, sys)
