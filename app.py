from flask import Flask,render_template,request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.exception import CustomException
from src.logger import logging

application=Flask(__name__)
app=application

@app.route("/")
def index():
    return render_template("mushroom.html")

@app.route("/predictions",methods=['GET','POST'])
def predict_class():
    if request.method=='GET':
        return render_template("mushroom.html")
    else:
        data=CustomData(
            cap_diameter=request.form.get('cap_diameter'),cap_shape=request.form.get('cap_shape'),
            cap_surface=request.form.get('cap_surface'),cap_color=request.form.get('cap_color'),
            does_bruise_or_bleed=request.form.get('does_bruise_or_bleed'),gill_attachment=request.form.get('gill_attachment'),
            gill_color=request.form.get('gill_color'),stem_height=request.form.get('stem_height'),
            stem_width=request.form.get('stem_width'),stem_color=request.form.get('stem_color'),
            has_ring=request.form.get('has_ring'),ring_type=request.form.get('ring_type'),
            habitat=request.form.get('habitat'),season=request.form.get('season'))
        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        pred_pipe=PredictPipeline()
        logging.info("prediction started")
        prediction=pred_pipe.predict(pred_df)
        if prediction[0]==1:
            results="Poisonous mushroom"
        elif prediction[0]==0:
            results="Edible mushroom"

        print("The mushroom is {}".format(results))


        return render_template('mushroom.html',results=results)

if __name__=="__main__":
    app.run()


