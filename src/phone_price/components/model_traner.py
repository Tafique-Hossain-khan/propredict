from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import os,sys
from src.utils import save_object

from sklearn.tree import DecisionTreeRegressor


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score


@dataclass 
class ModelTranerConfig:
    modelpath:str = os.path.join('artifacts\phone','model.pkl')


class ModelTraner:

    def __init__(self) -> None:
        self.model_traner_path = ModelTranerConfig()


    def model_traner(self,train_data,test_data):
        try:
            #split the datas
            X_train = train_data[:,:-1]
            y_train = train_data[:,-1]

            X_test = test_data[:,:-1]
            y_test = test_data[:,-1]

            dc = DecisionTreeRegressor()
            dc.fit(X_train,y_train)

            y_pred = dc.predict(X_test)
            r2 = r2_score(y_test,y_pred)

            save_object(self.model_traner_path.modelpath,dc)
            logging.info(r2)

            self
        except Exception as e:
            raise CustomException(e,sys)

        

        
