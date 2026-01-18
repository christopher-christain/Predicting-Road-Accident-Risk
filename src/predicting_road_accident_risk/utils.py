import os
import sys
import pickle
import numpy as np
import yaml

from src.predicting_road_accident_risk.exception import CustomException
from src.predicting_road_accident_risk.logger import logger


def save_object(file_path, obj):
    """
    Save a Python object as a pickle file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logger.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logger.error("Error occurred while saving object", exc_info=True)
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a Python object from a pickle file.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logger.info(f"Object loaded successfully from {file_path}")
        return obj

    except Exception as e:
        logger.error("Error occurred while loading object", exc_info=True)
        raise CustomException(e, sys)


def read_yaml_file(file_path):
    """
    Read a YAML configuration file and return its contents.
    """
    try:
        with open(file_path, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)

        logger.info(f"YAML file read successfully from {file_path}")
        return content

    except Exception as e:
        logger.error("Error occurred while reading YAML file", exc_info=True)
        raise CustomException(e, sys)


def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using common regression metrics.
    """
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error("Error occurred during model evaluation", exc_info=True)
        raise CustomException(e, sys)
