from src.laptop_price.components.data_transformation import DataTransformation,DataTransformationConfig
from src.laptop_price.components.model_traner import ModelTraner,ModelTranerConfig
from src.laptop_price.components.data_injection import DataInjection

from src.laptop_price.pipeline.prediction_pipeling import CustomInput,PredictPipeline 

from src.utils import load_object
from src.logger import logging
from src.exception import CustomException
import os
import sys

class TrainLaptop:
    def output(self,Company,	TypeName,	Ram,	Weight,	Touchscreen,	Ips,	ppi,	Cpu_brand,	HDD,	SSD,	Gpu_brand,	os):
        try:
            di = DataInjection()
            raw_data_path,train_data_path,test_data_path = di.initai_data_injection()


            dt = DataTransformation()
            #obj_path = dt.get_data_transformation_obj()
            train_data ,test_data =dt.initiate_data_transformation(train_data_path,test_data_path)

            mr = ModelTraner()
            mr.model_traner(train_data,test_data)



            #ci = CustomInput('Apple',	'Ultrabook',	8,	1.37,	0,	1	,226.983005,	'Intel Core i5',	0,	128,	'Intel',	'Mac')
            ci = CustomInput(Company,	TypeName,	Ram,	Weight,	Touchscreen,	Ips,	ppi,	Cpu_brand,	HDD,	SSD,	Gpu_brand,	os)
            df = ci.custom_dataset()
            logging.info(df)

            pred = PredictPipeline()
            prediction = pred.predict(df)
            return prediction
        

        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj = TrainLaptop()
    ans = obj.output('Apple',	'Ultrabook',	8,	1.37,	0,	1	,226.983005,	'Intel Core i5',	0,	128,	'Intel',	'Mac')
    logging.info(ans)
    '''
    try:
        di = DataInjection()
        raw_data_path,train_data_path,test_data_path = di.initai_data_injection()


        dt = DataTransformation()
        #obj_path = dt.get_data_transformation_obj()
        train_data ,test_data =dt.initiate_data_transformation(train_data_path,test_data_path)

        mr = ModelTraner()
        mr.model_traner(train_data,test_data)



        ci = CustomInput('Apple',	'Ultrabook',	8,	1.37,	0,	1	,226.983005,	'Intel Core i5',	0,	128,	'Intel',	'Mac')
        df = ci.custom_dataset()
        logging.info(df)

        pred = PredictPipeline()
        pred.predict(df)
        

    except Exception as e:
        raise CustomException(e,sys)
    '''


    
