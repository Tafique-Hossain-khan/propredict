from src.phone_price.components.data_transformation import DataTransformation,DataTransformationConfg
from src.phone_price.components.model_traner import ModelTraner,ModelTranerConfig
from src.phone_price.components.data_injection import DataInjection

from src.phone_price.pipeline.prediction_pipeline import CustomInput,PredictPipeline

from src.utils import load_object
from src.logger import logging
from src.exception import CustomException
import os
import sys

'''
class TraningPhone:
    
    def __init__(self,dispaly_res_type	,os,	processor_brand,	connectivity_feature,	ram,	internal_storgar	
                ,mobile_brand,	dispaly_size_in_inches,	back_camera,	front_camera,	Battery_Category) -> None:
        self.dispaly_res_type = dispaly_res_type
        self.os = os	
        self.processor_brand = processor_brand
        self.connectivity_feature = connectivity_feature
        self.ram = ram
        self.internal_storgar = internal_storgar
        self.mobile_brand = mobile_brand
        self.dispaly_size_in_inches = dispaly_size_in_inches
        self.back_camera = back_camera
        self.front_camera = front_camera
        self.Battery_Category = Battery_Category
        
    def output_phone(self,dispaly_res_type	,os,	processor_brand,	connectivity_feature,	ram,	internal_storgar	
                ,mobile_brand,	dispaly_size_in_inches,	back_camera,	front_camera,	Battery_Category):
        try:
            di = DataInjection()
            train_data_path,test_data_path = di.get_data()


            dt = DataTransformation()
            #obj_path = dt.get_data_transformation_obj()
            train_data ,test_data =dt.initiate_data_transformation(train_data_path,test_data_path)

            mr = ModelTraner()
            mr.model_traner(train_data,test_data)



            #ci = CustomInput('Full HD+',	'Android 11','Snapdragon',	4,	6	,128	,'vivo'	,'6',	48.0	,16.0,	'4000-5000')
            ci = CustomInput(dispaly_res_type	,os,	processor_brand,	connectivity_feature,	ram,	internal_storgar	
                ,mobile_brand,	dispaly_size_in_inches,	back_camera,	front_camera,	Battery_Category)
            df = ci.custome_dataset()
            logging.info(df)

            pred = PredictPipeline()
            prediction = pred.predict(df)
            return prediction
        

        except Exception as e:
            raise CustomException(e,sys)'''

if __name__ == "__main__":

    #obj = Tranin()
    #ns = obj.output('Full HD+',	'Android 11','Snapdragon',	4,	6	,128	,'vivo'	,'6',	48.0	,16.0,	'4000-5000')
    #ogging.info(ans)
    try:
        di = DataInjection()
        train_data_path,test_data_path = di.get_data()


        dt = DataTransformation()
        #obj_path = dt.get_data_transformation_obj()
        train_data ,test_data =dt.initiate_data_transformation(train_data_path,test_data_path)

        mr = ModelTraner()
        mr.model_traner(train_data,test_data)



        ci = CustomInput('Full HD+',	'Android 11','Snapdragon',	4,	6	,128	,'vivo'	,'6',	48.0	,16.0,	'4000-5000')
        df = ci.custome_dataset()
        logging.info(df)

        pred = PredictPipeline()
        pred.predict(df)
        

    except Exception as e:
        raise CustomException(e,sys)



    
