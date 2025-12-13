import sys
import os
import pandas as pd
from project.logger import logging
from project.exception import CustomException
from project.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocesser_obj_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Before loading the model")
            logging.info("Loading the model")

            model = load_object(model_path)
            preprocesser_obj = load_object(preprocesser_obj_path)

            print("After loading the model")
            logging.info("Loaded the model")

            processed_data = preprocesser_obj.transform(features)
            predicted_data = model.predict(processed_data)
            return predicted_data

        except Exception as e:
            raise CustomException(e,sys)

class CustomData:

    def __init__(self,writing_score: int, reading_score: int, 
    gender: str, race_ethnicity: str, parental_level_of_education: str, 
    lunch: str, test_preparation_course: str):

        self.writing_score = writing_score
        self.reading_score = reading_score
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course

    def get_data_as_dataframe(self):
         try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

         except Exception as e:
            raise CustomException(e,sys)
