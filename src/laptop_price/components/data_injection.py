import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.laptop_price.components.data_transformation import DataTransformation
from src.laptop_price.components.model_traner import ModelTraner
@dataclass
class DataInjectionConfig:

    raw_data_path:str = os.path.join('artifacts/laptop',"raw_data.csv")
    train_data_path:str = os.path.join('artifacts/laptop',"train_data.csv")
    test_data_path:str = os.path.join('artifacts/laptop',"test_data.csv")


class DataInjection:

    def __init__(self) -> None:
        self.data_path = DataInjectionConfig()

    
    def initai_data_injection(self):

        #read the data
        try:
            df = pd.read_csv('notebook/laptop_price/cleaned_laptop_data2.csv')
            logging.info(df.head())

            os.makedirs(os.path.dirname(self.data_path.raw_data_path),exist_ok=True)
                

            df.to_csv(self.data_path.raw_data_path,index=False,header=True)
            

            X_train,X_test = train_test_split(df,test_size=0.15,random_state=2)

            logging.info(X_train.shape)
            logging.info(X_test.shape)

            X_train.to_csv(self.data_path.train_data_path,index=False,header=True)
            X_test.to_csv(self.data_path.test_data_path,index=False,header=True)
            logging.info("Data Injection Done")
            return(
                self.data_path.raw_data_path,
                self.data_path.train_data_path,
                self.data_path.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)



'''
if __name__ == "__main__":
    di = DataInjection()
    raw_data,train_data,test_data = di.initai_data_injection()

    dt = DataTransformation()
    pre_obj = dt.get_data_transformation_obj()

    train_arr,test_arr = dt.initiate_data_transformation(train_data,test_data)

    mr = ModelTraner()
    mr.model_traner(train_arr,test_arr)
    '''