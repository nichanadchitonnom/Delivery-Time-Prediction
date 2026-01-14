import os
import sys
import pickle
from src.exception import CustomException
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def save_object(file_path, obj):
    """
    Save any python object as a pickle file
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a pickle object from disk
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(y_true, y_pred):
    """
    Evaluate regression model performance
    Returns MAE, RMSE, R2
    """
    try:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    except Exception as e:
        raise CustomException(e, sys)
