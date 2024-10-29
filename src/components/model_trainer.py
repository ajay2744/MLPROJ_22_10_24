
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder,OrdinalEncoder,LabelEncoder,FunctionTransformer
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import f1_score

import os
import sys

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj,evaluate_models

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","best_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,x,y):
        try:
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
            logging.info("Splitting the data into train and test data")
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(solver='newton-cg'),
                "Sgd": SGDClassifier(),
                "svc": SVC(),
                "k-nearest-neighbors": KNeighborsClassifier(),
            }

            model_report=evaluate_models(x_train,y_train,x_test,y_test,models)

            best_model_score=max(model_report.values())

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best model found")

            save_obj(self.model_trainer_config.trained_model_file_path,best_model)

            predict_test_best=best_model.predict(x_test)

            print("F1 score:",f1_score(y_test,predict_test_best))
            logging.info("Finally got the best model prediction")
                
        except Exception as e:
            raise CustomException(e,sys)
