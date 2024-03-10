import os,sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from src.utils import load_object

class PredictPipeline:

    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            logging.info('Prediction pipeline started')
            preprocessor_path = os.path.join('artifacts/phone','preprocessor.pkl')
            model_path = os.path.join('artifacts/phone','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            logging.info("object loaded")

            

            data_scaled = preprocessor.transform(features)
            logging.info("data scaling done")

            pred = model.predict(data_scaled)
            logging.info(f'The price is{pred}')
            return pred

        except Exception as e:
            raise CustomException(e,sys)


class CustomInput:

    def __init__(self, display_res_type, os, processor_brand, connectivity_feature, ram, internal_storage,
                mobile_brand, display_size_in_inches, back_camera, front_camera,Battery_Category):
        
        self.display_res_type = display_res_type
        self.os = os
        self.processor_brand = processor_brand
        self.connectivity_feature = connectivity_feature
        
        self.ram = ram
        self.internal_storage = internal_storage
        self.mobile_brand = mobile_brand
        self.display_size_in_inches = display_size_in_inches
        self.back_camera = back_camera
        self.front_camera = front_camera
        self.Battery_Category = Battery_Category

    def custome_dataset(self):
        
        try:
            attributes = {
                'dispaly_res_type': [self.display_res_type],
                'os': [self.os],
                'processor_brand': [self.processor_brand],
                'connectivity_feature': [self.connectivity_feature],
                'ram': [self.ram],
                'internal_storgar': [self.internal_storage],
                'mobile_brand': [self.mobile_brand],
                'dispaly_size_in_inches': [self.display_size_in_inches],
                'back_camera': [self.back_camera],
                'front_camera': [self.front_camera],
                'Battery_Category': [self.Battery_Category]
            }


            df = pd.DataFrame(attributes)
            return df

        except Exception as e:
            raise CustomException (e,sys)
                