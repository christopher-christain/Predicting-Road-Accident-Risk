from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.predicting_road_accident_risk.exception import CustomException
from src.predicting_road_accident_risk.logger import logger
import os

@dataclass

# defining where the data will be stored after ingestion
class dataingestionconfig:
    raw_data_pat: str=os.path.join("artifacts","raw_data.csv")
    train_data_path: str=os.path.join("artifacts", "train_data.csv")
    test_data_path: str = os.path.join("artifacts", "test_data.csv")