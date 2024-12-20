import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

#from src.components.model_trainer import ModelTrainerConfig
#from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    #train_data_path: str=os.path.join('artifacts',"train.csv")
    #test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\data.csv')
            logging.info('Read the dataset as dataframe')
            df=df.drop(columns=['stem-root','veil-type','veil-color','spore-print-color','stem-surface','gill-spacing'],axis=1)
            df=df.rename(columns={'cap-diameter':'cap_diameter','cap-shape':'cap_shape','cap-surface':'cap_surface','cap-color':'cap_color',
                                  'does-bruise-or-bleed':'does_bruise_or_bleed','gill-attachment':'gill_attachment','gill-color':'gill_color',
                                  'stem-height':'stem_height','stem-width':'stem_width','stem-color':'stem_color','has-ring':'has_ring',
                                  'ring-type':'ring_type'})

            #os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            #logging.info("Train test split initiated")
            #train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            #train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            #test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return (
                #self.ingestion_config.train_data_path,
                #self.ingestion_config.test_data_path
                self.ingestion_config.raw_data_path)
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    raw_data_path=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    x,y=data_transformation.initiate_data_transformation(raw_data_path=raw_data_path)

    model_trainer=ModelTrainer()
    model_trainer.initiate_model_trainer(x,y)

    