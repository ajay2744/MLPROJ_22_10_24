from src.exception import CustomException
from src.logger import logging

import dill
import pickle
import os
import sys

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)       
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        logging.info("Model saved")
    except Exception as e:
        CustomException(e,sys)

def evaluate_models(x_train, y_train,x_test,y_test,models):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            #para=param[list(models.keys())[i]]
            y_train_pred=cross_val_predict(model,x_train,y_train,cv=5)

            #gs = GridSearchCV(model,para,cv=3)
            #gs.fit(X_train,y_train)

            #model.set_params(**gs.best_params_)
            #model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            #y_train_pred = model.predict(X_train)
            model.fit(x_train,y_train)

            y_test_pred = model.predict(x_test)

            train_model_score = f1_score(y_train, y_train_pred)

            test_model_score = f1_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        logging.info("Models report dictionary generated")

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    


