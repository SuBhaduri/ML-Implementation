# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import json
import joblib
import numpy as np
#from azureml.core.model import Model

print("subha 0")
# Called when the service is loaded
def init():
    global model
    # Get the path to the registered model file and load it
    model = joblib.load("Kaggle_Titanic_RF_Azure_Deploy.pkl")
    print("subha 1")

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    # Get a prediction from the model
    predictions = model.predict(data)
    # Return the predictions as any JSON serializable format
    print("subha 2")
    return predictions.tolist()

