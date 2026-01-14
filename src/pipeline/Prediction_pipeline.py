import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor = load_object("Artifacts/preprocessor.pkl")
            model = load_object("Artifacts/model.pkl")

            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        Distance_km: float,
        Preparation_Time_min: float,
        Courier_Experience_yrs: float,
        Weather: str,
        Traffic_Level: str,
        Time_of_Day: str,
        Vehicle_Type: str,
        Is_Peak_Hour: int,
        Peak_Score: int,
        Distance_per_Experience: float,
    ):
        self.Distance_km = Distance_km
        self.Preparation_Time_min = Preparation_Time_min
        self.Courier_Experience_yrs = Courier_Experience_yrs
        self.Weather = Weather
        self.Traffic_Level = Traffic_Level
        self.Time_of_Day = Time_of_Day
        self.Vehicle_Type = Vehicle_Type
        self.Is_Peak_Hour = Is_Peak_Hour
        self.Peak_Score = Peak_Score
        self.Distance_per_Experience = Distance_per_Experience

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Distance_km": [self.Distance_km],
                "Preparation_Time_min": [self.Preparation_Time_min],
                "Courier_Experience_yrs": [self.Courier_Experience_yrs],
                "Weather": [self.Weather],
                "Traffic_Level": [self.Traffic_Level],
                "Time_of_Day": [self.Time_of_Day],
                "Vehicle_Type": [self.Vehicle_Type],
                "Is_Peak_Hour": [self.Is_Peak_Hour],
                "Peak_Score": [self.Peak_Score],
                "Distance_per_Experience": [self.Distance_per_Experience],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
