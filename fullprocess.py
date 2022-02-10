

from xxlimited import new
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting

import pandas as pd
import numpy as np
import os
import json



#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']  # practicedata folder
output_folder_path = config['output_folder_path']  #ingesteddata
prod_deployment_path = config['prod_deployment_path'] #production_deployment
test_data_path = os.path.join(config['test_data_path']) #testdata


##################Check and read new data
#first, read ingestedfiles.txt

output_dir_fullname = os.path.join(os.getcwd(), output_folder_path)
with open(os.path.join(output_dir_fullname, 'ingestedfiles.txt'), 'rb') as f:
    stored_filenames = [x.decode('utf8').strip() for x in f.readlines()]

current_filenames = (x for x in os.listdir(os.path.join(os.getcwd(), input_folder_path)) if x[-4:] == '.csv')

new_unique_files = []
for file in current_filenames:
    if file not in stored_filenames:
        new_unique_files.append(file)

print(new_unique_files)
if new_unique_files:
    print("new files")
    ingestion.merge_multiple_dataframe()

print("done")


#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt



##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model







