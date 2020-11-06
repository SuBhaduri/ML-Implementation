# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
#from azureml.core.model import Model

app = Flask(__name__)
print("subha 0")

model = joblib.load("Kaggle_Titanic_RF_Azure_Deploy.pkl")
print("subha 1")
# Called when the service is loaded
#def init():
#    global model
    # Get the path to the registered model file and load it

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

# Called when a request is received
@app.route('/predict_api',methods=['POST'])
def predict_api(raw_data):
#def run(raw_data):
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    # Get a prediction from the model
    predictions = model.predict(data)
    # Return the predictions as any JSON serializable format
    print("subha 2")
    return predictions.tolist()

