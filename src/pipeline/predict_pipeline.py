import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path="artifacts\best_model.pkl"
            preprocessor_path="artifacts\preprocessor.pkl"
            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)
            data_processed=preprocessor.transform(features)
            preds=model.predict(data_processed)

            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                cap_diameter,cap_shape,cap_surface,cap_color,does_bruise_or_bleed,gill_attachment,gill_color,stem_height,stem_width,
                stem_color,has_ring,ring_type,habitat,season):
        
        
        self.cap_diameter=cap_diameter
        self.cap_shape=cap_shape
        self.cap_surface=cap_surface
        self.cap_color=cap_color
        self.does_bruise_or_bleed=does_bruise_or_bleed
        self.gill_attachment=gill_attachment
        self.gill_color=gill_color
        self.stem_height=stem_height
        self.stem_width=stem_width
        self.stem_color=stem_color
        self.has_ring=has_ring
        self.ring_type=ring_type
        self.habitat=habitat
        self.season=season

    def get_data_as_data_frame(self):
        try:
            custome_data_input_dict={
            "cap_diameter":[self.cap_diameter],
            "cap_shape":[self.cap_shape],
            "cap_surface":[self.cap_surface],
            "cap_color":[self.cap_color],
            "does_bruise_or_bleed":[self.does_bruise_or_bleed],
            "gill_attachment":[self.gill_attachment],
            "gill_color":[self.gill_color],
            "stem_height":[self.stem_height],
            "stem_width":[self.stem_width],
            "stem_color":[self.stem_color],
            "has_ring":[self.has_ring],
            "ring_type":[self.ring_type],
            "habitat":[self.habitat],
            "season":[self.season],
            }

            return pd.DataFrame(custome_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)


