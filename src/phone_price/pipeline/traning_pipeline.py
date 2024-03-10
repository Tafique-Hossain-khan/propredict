from src.phone_price.components.data_transformation import DataTransformation,DataTransformationConfg
from src.phone_price.components.model_traner import ModelTraner,ModelTranerConfig
from src.phone_price.components.data_injection import DataInjection

from src.phone_price.pipeline.prediction_pipeline import CustomInput,PredictPipeline

from src.utils import load_object
from src.logger import logging
from src.exception import CustomException
import os
import sys
if __name__ == "__main__":

    try:
        di = DataInjection()
        train_data_path,test_data_path = di.get_data()


        dt = DataTransformation()
        #obj_path = dt.get_data_transformation_obj()
        train_data ,test_data =dt.initiate_data_transformation(train_data_path,test_data_path)

        mr = ModelTraner()
        mr.model_traner(train_data,test_data)



        ci = CustomInput('Full HD+',	'Android 11','Snapdragon',	'4',	6	,128	,'vivo'	,'6',	48.0	,16.0,	'4000-5000')
        df = ci.custome_dataset()
        logging.info(df)

        pred = PredictPipeline()
        pred.predict(df)
        

    except Exception as e:
        raise CustomException(e,sys)



    
