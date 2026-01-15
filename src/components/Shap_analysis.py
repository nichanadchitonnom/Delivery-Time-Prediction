import os
import sys
import shap
import pandas as pd
import matplotlib.pyplot as plt

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.Feature_engineering import FeatureEngineering

def run_shap_analysis():
    try:
        logging.info("Starting SHAP analysis")

        # ===============================
        # Paths
        # ===============================
        artifacts_dir = "artifacts"

        model_path = os.path.join(artifacts_dir, "model.pkl")
        preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
        data_path = "Notebook_Experiments/Data/Food_Delivery_Times.csv"

        # ===============================
        # Load objects
        # ===============================
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)

        df = pd.read_csv(data_path)

        # ===============================
        # Split X / y
        # ===============================
        target_column = "Delivery_Time_min"   
        X = df.drop(columns=[target_column])

        # ===== Feature Engineering (สำคัญมาก) =====
        fe = FeatureEngineering()
        X = fe.add_peak_features(X)

        # ===== Transform =====
        X_transformed = preprocessor.transform(X)
        # ===============================
        # Transform features
        # ===============================
        X_transformed = preprocessor.transform(X)

        feature_names = preprocessor.get_feature_names_out()

        X_transformed_df = pd.DataFrame(
            X_transformed,
            columns=feature_names
        )

        logging.info("Data transformed successfully")

        # ===============================
        # SHAP explainer
        # ===============================
        explainer = shap.Explainer(
            model.predict,
            X_transformed_df
        )

        shap_values = explainer(X_transformed_df)

        # ===============================
        # SHAP plots
        # ===============================
        shap.summary_plot(
            shap_values,
            X_transformed_df,
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_dir, "shap_summary.png"))
        plt.close()

        shap.summary_plot(
            shap_values,
            X_transformed_df,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_dir, "shap_feature_importance.png"))
        plt.close()

        logging.info("SHAP analysis completed successfully")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_shap_analysis()
