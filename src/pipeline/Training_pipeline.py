from src.components.Data_ingestion import DataIngestion
from src.components.Feature_engineering import FeatureEngineering
from src.components.Data_transformation import DataTransformation
from src.components.Model_trainer import ModelTrainer

# 1️⃣ Data Ingestion
obj = DataIngestion()
train_data, test_data = obj.initiate_data_ingestion()

# 2️⃣ Feature Engineering (สร้าง Peak_Score, Is_Peak_Hour)
fe = FeatureEngineering()
train_data = fe.add_peak_features(train_data)
test_data = fe.add_peak_features(test_data)

# 3️⃣ Data Transformation (scaling + encoding)
data_transformation = DataTransformation()
train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
    train_data,
    test_data
)

# 4️⃣ Model Training
modeltrainer = ModelTrainer()
print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
