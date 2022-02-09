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
@app.route("/prediction", methods=['GET','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    filename = request.args.get('filename')
    predictions = model_predictions(filename)
    intro_str = f'Model predictions from dataset {filename}: '
    predictions_str = ' '.join(map(str, predictions))
    summary_str = intro_str + predictions_str    

    return summary_str

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats1():
    #check the score of the deployed model
    filename = request.args.get('filename')
    f1_score = score_model(filename)
    return str(f1_score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats2():        
    #check means, medians, and modes for each column
    filename = request.args.get('filename')
    summary = dataframe_summary(filename)
    summary_str = '\n'.join(map(str, summary))
    return summary_str

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats3():        
    #check timing and percent NA values
    exec_time = execution_time()
    exec_time_str = f"Ingestion timing {exec_time[0]}. Training timing {exec_time[1]}"
    return exec_time_str

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
