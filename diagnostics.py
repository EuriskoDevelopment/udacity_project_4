
import pandas as pd
import numpy as np
import timeit
import os
import subprocess
import json
import pickle
from common import load_data, numeric_predictors
from training import train_model
from ingestion import merge_multiple_dataframe

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

##################Function to get model predictions
def model_predictions(filename):
    #read the deployed model and a test dataset, calculate predictions
    X, y = load_data(os.path.join(os.getcwd(), test_data_path, filename))
    
    with open( os.path.join(os.getcwd(), prod_deployment_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)

    predictions = model.predict(X)

    return predictions

##################Function to get summary statistics
def dataframe_summary(filename):
    #calculate summary statistics here
    data = pd.read_csv(os.path.join(os.getcwd(), dataset_csv_path, filename))

    mean = data.mean(axis=0)
    std = data.std(axis=0)
    median = data.median(axis=0)
    summary = []
    for col in numeric_predictors:
        mean_num = mean[col]
        std_num = std[col]
        median_num = median[col]
        summary.append(f"Statistics for {col}. Mean {mean_num}. Std {std_num}. Median {median_num}")

    return summary

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    training_starttime = timeit.default_timer()
    os.system('python3 training.py')
    training_timing=timeit.default_timer() - training_starttime

    ingestion_starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingestion_timing=timeit.default_timer() - ingestion_starttime

    result = [ingestion_timing, training_timing]
    return result

##################Function to check dependencies
def outdated_packages_list():
    summary = []

    pip_list_lines = subprocess.run(['pip', 'list','--outdated'], stdout=subprocess.PIPE).stdout.splitlines()    
    outdated_modules = (x.decode('utf8').strip().split() for x in pip_list_lines)

    with open( os.path.join(os.getcwd(), 'requirements.txt'), 'rb') as file:
        req_modules = [x.decode('utf8').strip() for x in file.readlines()]

    for req_mod in req_modules:
        req_mod_info = req_mod.split("==")
        for outdate_mod in outdated_modules:
            if req_mod_info[0] == outdate_mod[0]:
                print(outdate_mod[0] + "  " + req_mod_info[0])
                summary.append(f"{outdate_mod[0]} {outdate_mod[1]} {outdate_mod[2]}")
                break

    print(summary)
    return summary 


if __name__ == '__main__':
    model_predictions('testdata.csv')
    dataframe_summary('finaldata.csv')
    execution_time()
    outdated_packages_list()





    
