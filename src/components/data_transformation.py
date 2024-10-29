import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.pipeline import make_pipeline
from feature_engine.outliers import Winsorizer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")
    

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_trasformer_object(self,raw_data_path):
        try:
            df=pd.read_csv(raw_data_path)
            logging.info("Read raw data")
            x=df.drop(columns=['class'],axis=1)
            
            
            num_cols=x.select_dtypes(exclude='object').columns
            cat_cols=x.select_dtypes(include='object').columns        
          
            num_pipe=make_pipeline(SimpleImputer(strategy='mean'),Winsorizer(capping_method='gaussian'),StandardScaler())
            cat_pipe=make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(sparse_output=False))
            preprocessor=ColumnTransformer([('num_pipeline',num_pipe,num_cols),('cat_pipeline',cat_pipe,cat_cols)])
            logging.info("Data preprocessor created")


            return preprocessor
                
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,raw_data_path):
        try:
            df=pd.read_csv(raw_data_path)
            x=df.drop(columns=['class'])
            y=df['class']
            lable_encoder=LabelEncoder()
            y=lable_encoder.fit_transform(y)
            logging.info("Data split into dependent and independent variables")

            preprocessor_obj=self.get_data_trasformer_object(raw_data_path)
            logging.info("Fit and transform with column transformer starting")
            x=preprocessor_obj.fit_transform(x)
            

            save_obj(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessor_obj)

            return x,y 
        except Exception as e:
            CustomException(e,sys)
            


