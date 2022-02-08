from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from scoring import score_model
from diagnostics import dataframe_summary, execution_time, model_predictions
#import create_prediction_model
#import predict_exited_from_saved_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None

#def readpandas(filename):
#    thedata=pd.read_csv(os.path.join(os.getcwd(), test_data_path, filename)
#    return thedata

@app.route('/')
def index():
    return "Hello"

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    filename = request.args.get('filename')
    prediction= model_predictions(filename)
    return str(prediction)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats1():
    filename = request.args.get('filename')
    f1_score = score_model(filename)
    #check the score of the deployed model
    return f1_score

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats2():        
    filename = request.args.get('filename')
    summary = dataframe_summary(filename)
    #check means, medians, and modes for each column
    return summary

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats3():        
    #filename = request.args.get('filename')
    exec_time = execution_time()
    #check timing and percent NA values
    exec_time_str = f"Ingestion timing {exec_time[0]}. Training timing {exec_time[1]}"
    return exec_time_str

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
