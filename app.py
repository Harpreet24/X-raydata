from flask import Flask, render_template, request,redirect,url_for
import cv2
import os
import glob
import sys
import re
import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np


app = Flask(__name__)
model_path='models.h5'

model=load_model(model_path)
model._make_predict_function()

def model_predict(img_path,model):
    img=img.load_img(img_path,target_size=(100,100))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)
    preds=model.predict(x)
    return preds

@app.route('/',method=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method=="POST":
        f = request.files['file']
        basepath = os.path.dirname(__file__)    
        file_path = os.path.join(basepath, 'uploads',secure_filename(f.filename))
        f.save(file_path)
        pred = model_predict(file_path,model)
        pred_class = decode_predictions(pred,top=1)
        result = str(pred_class[0][0][1])
        return result

    return None




if __name__ == '__main__':
    app.run(debug=True)



