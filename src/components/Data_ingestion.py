# src/components/data_ingestion.py

import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("🚚 Data Ingestion started")

        try:
            # 1. Read delivery dataset
            df = pd.read_csv("Notebook_Experiments/Data/Food_Delivery_Times.csv")
            logging.info("Delivery dataset loaded")

            # 2. Create artifacts folder
            os.makedirs("artifacts", exist_ok=True)

            # 3. Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")

            # 4. Train-test split
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # 5. Save train & test data
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Train-test split completed")

            return train_df, test_df

        except Exception as e:
            logging.error("Error in data ingestion")
            raise CustomException(e, sys)
