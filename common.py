import pandas as pd
import numpy as np
import pickle
import os
import json

with open('config.json','r') as f:
    config = json.load(f) 

numeric_predictors = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
target = 'exited'
dataset_csv_path = os.path.join(config['output_folder_path']) 


def load_data(data_path):
    trainingdata=pd.read_csv(data_path)
    X=trainingdata.loc[:, numeric_predictors].values.reshape(-1, len(numeric_predictors))
    y=trainingdata[target].values.reshape(-1, 1).ravel()

    return (X, y)


