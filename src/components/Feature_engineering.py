import sys
from src.exception import CustomException
from src.logger import logging


class FeatureEngineering:
    def __init__(self):
        pass

    def add_peak_features(self, df):
        """
        Create Peak_Score and Is_Peak_Hour features
        based on Time_of_Day and Traffic_Level
        """
        try:
            df = df.copy()
            logging.info("Starting feature engineering: Peak features")

            time_score = {
                'Morning': 2,
                'Afternoon': 2,
                'Evening': 2,
                'Night': 1
            }

            traffic_score = {
                'Low': 0,
                'Medium': 1,
                'High': 2
            }

            df['Peak_Score'] = (
                df['Time_of_Day'].map(time_score) +
                df['Traffic_Level'].map(traffic_score)
            )

            df['Is_Peak_Hour'] = (df['Peak_Score'] >= 3).astype(int)

            logging.info("Peak features created successfully")

            return df

        except Exception as e:
            raise CustomException(e, sys)
