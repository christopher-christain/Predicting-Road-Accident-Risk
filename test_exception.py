import sys
from src.predicting_road_accident_risk.exception import CustomException
from src.predicting_road_accident_risk.logger import logger

try:
    1 / 0
except Exception as e:
    logger.error("Error occurred", exc_info=True)
    raise CustomException(e, sys)
