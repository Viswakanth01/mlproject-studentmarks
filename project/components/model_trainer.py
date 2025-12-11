import os
import sys
from project.logger import logging
from project.exception import CustomException
from project.components.data_transformation import DataTransformationConfig, DataTransformation
from project.utils import save_obj, evaluate_models
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join("artifacts", "model.pkl")

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        logging.info("Model Trainer Initialized")

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models: dict={
                'LinearRegression': LinearRegression(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor()
            }

            logging.info("Training for best model")

            model_report: dict=evaluate_models(X_train, y_train, X_test, y_test, models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.75:
                raise CustomException("Model is a bit inaccurate")

            logging.info("Best model Found")

            save_obj(file_path= self.model_trainer_config.trained_model_file_path, obj=best_model)

            logging.info("Best model saved")

            predicted = best_model.predict(X_test)
            r_square=r2_score(y_test, predicted)

            return r_square
            logging.info("Model Trainer completed.")

        except Exception as e:
            raise CustomException(e, sys)
        

