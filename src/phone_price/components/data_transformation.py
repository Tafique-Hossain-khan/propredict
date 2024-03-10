from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import os,sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object


@dataclass
class DataTransformationConfg:
    preprocessor_path:str = os.path.join('artifacts\phone','preprocessor.pkl')


class DataTransformation:

    def __init__(self) -> None:
        self.path = DataTransformationConfg()



    def get_data_transformation_obj(self):

        try:
            df = pd.read_csv('notebook\data\cleaned_phone_data1.csv')
        
            cat_col = df.select_dtypes(include=['object']).columns
            logging.info(cat_col)

            cat_pip = Pipeline([
            ('ohe',OneHotEncoder(drop='first'))
            ])

        
            preprocessor = ColumnTransformer([
                ('cat',cat_pip,cat_col),
                
            ],remainder='passthrough')
            

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_data,test_data):

        try:
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            #spliting traning dataset
            input_feature_train_df = train_df.drop(columns=['price'],axis='columns')
            traget_feature_train_df = train_df['price']

            #logging.info(input_feature_train_df.columns)

            #spliting test dataset
            input_feature_test_df = test_df.drop(columns=['price'],axis='columns')
            traget_feature_test_df = test_df['price']

            #load the preprocessor object
            preprocessor = self.get_data_transformation_obj()

            X_train_encoded = preprocessor.fit_transform(input_feature_train_df)
            X_test_encoded = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[X_train_encoded,np.array(traget_feature_train_df)]
            test_arr = np.c_[X_test_encoded,np.array(traget_feature_test_df)]

            save_object(self.path.preprocessor_path,preprocessor)
            logging.info("preprocessor obj saved in artifacts")
            return(
                train_arr,
                test_arr
            )
        
        except Exception as e:
            raise CustomException(e,sys)



