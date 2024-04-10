import pandas as pd
from src.exception import CustomException
import os,sys
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    
    def __init__(self) -> None:
        pass
        
    def predict(self,features):

        try:
            logging.info('Prediction pipeline started')
            preprocessor_path = os.path.join('artifacts/laptop','preprocessor_laptop.pkl')
            model_path = os.path.join('artifacts/laptop','model_laptop.pkl')

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
        
class CustomInputLaptop:
        def __init__(self, company, TypeName, Ram, Weight, Touchscreen, IPS,ppi, Cpu_brand, HDD, SSD, Gpu_brand, os):
            self.company = company
            self.TypeName = TypeName
            self.Ram = Ram
            self.Weight = Weight
            self.Touchscreen = Touchscreen
            self.IPS = IPS
            self.ppi = ppi
            self.Cpu_brand = Cpu_brand
            self.HDD = HDD
            self.SSD = SSD
            self.Gpu_brand = Gpu_brand
            self.os = os

        def custom_dataset(self):
            
            try:
                attribute =     {
                    'Company': [self.company],
                    'TypeName': [self.TypeName],
                    'Ram': [self.Ram],
                    'Weight': [self.Weight],
                    'Touchscreen': [self.Touchscreen],
                    'IPS ': [self.IPS],
                    'ppi': [self.ppi],
                    'Cpu_brand': [self.Cpu_brand],
                    'HDD': [self.HDD],
                    'SSD': [self.SSD],
                    'Gpu_brand': [self.Gpu_brand],
                    'os': [self.os]
                }

                df = pd.DataFrame(attribute)
                return df
            
            except Exception as e:

                raise CustomException(e,sys)
            
            