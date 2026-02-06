import os
import sys
import joblib
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.predicting_road_accident_risk.logger import logging
from src.predicting_road_accident_risk.exception import CustomException
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function creates and returns the preprocessing pipeline
        """
        try:
            numerical_features = [
                "num_lanes",
                "curvature",
                "speed_limit",
                "lane_complexity"
            ]

            categorical_features = [
                "road_type",
                "lighting",
                "weather",
                "time_of_day",
                "speed_risk_band"
            ]

            boolean_features = [
                "road_signs_present",
                "public_road",
                "holiday",
                "school_season",
                "high_curvature",
                "poor_visibility"
            ]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numerical_features),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
                    ("bool", "passthrough", boolean_features)
                ],
                remainder='drop'  # Explicitly drop any other columns
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, df):
        """
        Applies transformations and saves the preprocessor
        """
        try:
            logging.info("Starting data transformation")
            logging.info(f"Input dataframe shape: {df.shape}")
            logging.info(f"Columns in dataframe: {df.columns.tolist()}")

            # Drop ID column
            df = df.drop(columns=["id"], errors="ignore")

            target_column = "accident_risk"
            
            # Verify target column exists
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataframe")
            
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Convert boolean columns to int (0/1)
            boolean_features = [
                "road_signs_present",
                "public_road",
                "holiday",
                "school_season",
                "high_curvature",
                "poor_visibility"
            ]
            
            # Check if boolean features exist
            missing_features = [col for col in boolean_features if col not in X.columns]
            if missing_features:
                logging.warning(f"Missing boolean features: {missing_features}")
            
            # Convert only existing boolean features
            existing_bool_features = [col for col in boolean_features if col in X.columns]
            X[existing_bool_features] = X[existing_bool_features].astype(int)

            logging.info("Getting preprocessor object")
            preprocessor = self.get_data_transformer_object()

            logging.info("Fitting and transforming data")
            X_transformed = preprocessor.fit_transform(X)
            
            logging.info(f"Transformed data shape: {X_transformed.shape}")

            # Train-test split with stratification if target is categorical
            X_train, X_test, y_train, y_test = train_test_split(
                X_transformed, y, test_size=0.2, random_state=42, stratify=y
            )

            logging.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

            # Save preprocessor
            os.makedirs(
                os.path.dirname(self.transformation_config.preprocessor_obj_file_path), 
                exist_ok=True
            )
            joblib.dump(preprocessor, self.transformation_config.preprocessor_obj_file_path)

            logging.info(f"Preprocessor saved at: {self.transformation_config.preprocessor_obj_file_path}")

            return (
                X_train, 
                X_test, 
                y_train, 
                y_test,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)