import sys
import pandas as pd

from src.predicting_road_accident_risk.logger import logging
from src.predicting_road_accident_risk.exception import CustomException

from src.predicting_road_accident_risk.components.data_ingestion import DataIngestion
from src.predicting_road_accident_risk.components.data_transformation import DataTransformation
#from src.predicting_road_accident_risk.components.model_trainer import ModelTrainer


def main():
    try:
        logging.info("Starting end-to-end ML pipeline")

        # 1️ Load RAW data (not preprocessed)
        df = pd.read_csv("notebooks/data/raw/train.csv")
        logging.info("Raw dataset loaded successfully")

        # 2️ Data Ingestion
        ingestion = DataIngestion()
        raw_df = ingestion.initiate_data_ingestion(df)

        # 3️ Data Transformation
        transformation = DataTransformation()
        X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(raw_df)

      

    except Exception as e:
        logging.error("Pipeline failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
