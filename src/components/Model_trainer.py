import os
import sys
import numpy as np
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("Artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test arrays")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # ===============================
            # Models + Hyperparameter Space
            # ===============================
            models = {
                "Linear Regression": (
                    LinearRegression(),
                    {}
                ),

                "Decision Tree": (
                    DecisionTreeRegressor(random_state=42),
                    {
                        "max_depth": [None, 5, 10, 20],
                        "min_samples_split": [2, 5, 10]
                    }
                ),

                "Random Forest": (
                    RandomForestRegressor(random_state=42),
                    {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [None, 5, 10],
                        "min_samples_split": [2, 5]
                    }
                ),

                "XGBoost": (
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=42
                    ),
                    {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [4, 6, 8],
                        "learning_rate": [0.03, 0.05, 0.1],
                        "subsample": [0.7, 0.8, 1.0]
                    }
                ),

                "LightGBM": (
                    LGBMRegressor(random_state=42),
                    {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [-1, 6, 10],
                        "learning_rate": [0.03, 0.05, 0.1],
                        "subsample": [0.7, 0.8, 1.0]
                    }
                )
            }

            best_model = None
            best_r2 = -1

            # ===============================
            # Fair Model Comparison
            # ===============================
            for name, (model, params) in models.items():
                logging.info(f"Tuning model: {name}")

                if params:
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=params,
                        n_iter=20,
                        scoring="r2",
                        cv=3,
                        random_state=42,
                        n_jobs=-1
                    )
                    search.fit(X_train, y_train)
                    final_model = search.best_estimator_
                else:
                    final_model = model.fit(X_train, y_train)

                y_pred = final_model.predict(X_test)
                mae, rmse, r2 = evaluate_model(y_test, y_pred)

                logging.info(
                    f"{name} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}"
                )

                if r2 > best_r2:
                    best_r2 = r2
                    best_model = final_model

            if best_model is None:
                raise CustomException("No suitable model found", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"Best model saved with R2 score: {best_r2:.3f}")

            return best_r2

        except Exception as e:
            raise CustomException(e, sys)
