import os
import sys
import pandas as pd
from project.logger import logging
from project.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from project.components.data_transformation import DataTransformationConfig, DataTransformation
from project.components.model_trainer import ModelTrainerConfig, Modeltrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")

        try:
            df=pd.read_csv("data\stud.csv")
            logging.info("Read the dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info("Directory created")

            df.to_csv(self.ingestion_config.raw_data_path, index = False, header=True)
            logging.info("Moved the dataset to directory")

            logging.info("Train Test split initiated")

            train_data, test_data = train_test_split(df, test_size=0.25, random_state=42)
            
            logging.info("Moving train and test data to repective folders")
            
            train_data.to_csv(self.ingestion_config.train_data_path, index = False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index = False, header=True)

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

            logging.info("Data ingestion completed")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ =="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = Modeltrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))


            
            

            
