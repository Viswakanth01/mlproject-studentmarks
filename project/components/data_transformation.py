import os
import sys
import pandas as pd
import numpy as np
from project.logger import logging
from project.exception import CustomException
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from project.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_to_transform(self):

        try:
            logging.info("Extracting data from ingestion")

            num_columns = ['writing_score', 'reading_score']
            cat_columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]


            numeric_pipe = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('sc', StandardScaler())
                ]
            )

            categ_pipe = Pipeline(
                steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('one_hot', OneHotEncoder()),
                ('sc', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Pipeline created for numercial columns: {num_columns}")
            logging.info(f"Pipeline created for categorical columns: {cat_columns}")

            logging.info("preocess initialized")

            preprocess = ColumnTransformer(
                [
                ('numeric_pipe', numeric_pipe, num_columns),
                ('categ_pipe', categ_pipe, cat_columns)
                ]
            )
            logging.info("Pipeline created")

            return preprocess

        except Exception as e: 
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info("initiated data transformation")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("train and test Dataframes created")

            preprocess_obj = self.get_data_to_transform()

            logging.info("Preprocessing object created")
            
            target_columns = 'math_score'

            input_feature_train_df = train_df.drop(columns=[target_columns])
            input_feature_test_df = test_df.drop(columns=[target_columns])

            logging.info("input feature created to be processed")

            processed_input_feature_train_df_arr = preprocess_obj.fit_transform(input_feature_train_df)
            processed_input_feature_test_df_arr = preprocess_obj.transform(input_feature_test_df)

            logging.info("Data transformation completed")

            train_arr= np.c_[processed_input_feature_train_df_arr, np.array(train_df[target_columns])]

            test_arr= np.c_[processed_input_feature_test_df_arr, np.array(test_df[target_columns])]

            logging.info("processed data split into train and test arrays")


            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocess_obj
            )
            logging.info("saved as pickle file and data transformation is completed")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)




            




            




    
