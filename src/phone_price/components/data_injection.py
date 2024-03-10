import os
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import pandas as pd
import sys

from sklearn.model_selection import train_test_split

@dataclass
class DataInjectionConfig:
    
    raw_phone_data_path:str = os.path.join('artifacts\phone','raw_data.csv')
    train_data_path:str = os.path.join('artifacts\phone','train_data.csv')
    test_data_path:str = os.path.join('artifacts\phone','test_data.csv')


class DataInjection:
    
    def __init__(self) -> None:
        self.data_injection_config = DataInjectionConfig()


    def get_data(self):
        logging.info('Started')

        try:

            df = pd.read_csv('notebook\data\cleaned_phone_data1.csv')
            logging.info(df.shape)
            logging.info(df.head())


            os.makedirs(os.path.dirname(self.data_injection_config.raw_phone_data_path),exist_ok=True)
            
            df.to_csv(self.data_injection_config.raw_phone_data_path,index=False,header=True)
            logging.info(df.columns)
        

            train_data,test_data = train_test_split(df,test_size=0.3,random_state=42)
            

            train_data.to_csv(self.data_injection_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.data_injection_config.test_data_path,index=False,header=True)
            logging.info(train_data.shape)
            logging.info(test_data.shape)
            logging.info("Data Injection Done")
            return(
                self.data_injection_config.train_data_path,
                self.data_injection_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)




        
    
