import os
from src.exception import CustomException
from src.logger import logging
import sys
import numpy as np
import pandas as pd
import pickle 

from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(obj,f)


    except Exception as e:
        raise CustomException(e,sys)
    

    
def load_object(file_path):
    try:
        with open(file_path,'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e,sys)
