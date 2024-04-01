from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import os,sys
import pandas as pd                 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.utils import save_object,load_object
import numpy as np


@dataclass
class DataTransformationConfig:
    preprocessor_path_config:str = os.path.join('artifacts/laptop','preprocessor_laptop.pkl')

class DataTransformation:

    def __init__(self) -> None:
        self.preprocessor_path = DataTransformationConfig()


    def get_data_transformation_obj(self):


        try:
            #df = pd.read_csv('artifacts/laptop/raw_data.csv')
            df = pd.read_csv('notebook/laptop_price/cleaned_laptop_data2.csv')
            #df.drop(columns=['Unnamed: 0','ppi'],axis='columns',inplace=True)
            logging.info(df.columns)
            cat_col = df.select_dtypes(include=[object]).columns

            step1 = ColumnTransformer(transformers=[
                ('col_tnf',OneHotEncoder(drop='first',sparse_output=False),cat_col) #[0,1,6,9,10]
            ],remainder='passthrough')

            pipe = Pipeline([
                ('step1',step1),
                #('step2',step2)
            ])
            logging.info("Got the data transformation object")
            return step1
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_data,test_data):

        try:
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)
            #pipe = load_object("artifacts\laptop\pipe.pkl")

            logging.info(train_df.shape)
            
            #train_df.drop(columns=['Unnamed: 0','ppi'],axis='columns',inplace=True)
            #test_df.drop(columns=['Unnamed: 0','ppi'],axis='columns',inplace=True)
            logging.info("TEst data columns")
            logging.info(test_df.columns)

            #spliting traning dataset
            input_feature_train_df = train_df.drop(columns=['Price'],axis='columns')
            traget_feature_train_df = train_df['Price']

            #logging.info(input_feature_train_df.columns)

            #spliting test dataset
            input_feature_test_df = test_df.drop(columns=['Price'],axis='columns')
            traget_feature_test_df = test_df['Price']
            logging.info("input feature train df")
            logging.info(input_feature_test_df.columns)

            preprocessor = self.get_data_transformation_obj()

            X_train_encoded = preprocessor.fit_transform(input_feature_train_df)
            X_test_encoded = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[X_train_encoded,np.array(traget_feature_train_df)]
            test_arr = np.c_[X_test_encoded,np.array(traget_feature_test_df)]

            save_object(self.preprocessor_path.preprocessor_path_config,preprocessor)
            logging.info("preprocessor obj saved in artifacts")

            return(
                train_arr,
                test_arr
            )

        except Exception as e:
            raise CustomException(e,sys)
        




