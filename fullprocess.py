

from xxlimited import new
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import apicalls

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
model_path = os.path.join(config['output_model_path']) #practicemodels


##################Check and read new data
#first, read ingestedfiles.txt

output_dir_fullname = os.path.join(os.getcwd(), output_folder_path)
with open(os.path.join(output_dir_fullname, 'ingestedfiles.txt'), 'rb') as f:
    stored_filenames = [x.decode('utf8').strip() for x in f.readlines()]

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
current_filenames = (x for x in os.listdir(os.path.join(os.getcwd(), input_folder_path)) if x[-4:] == '.csv')

new_unique_files = []
for file in current_filenames:
    if file not in stored_filenames:
        new_unique_files.append(file)

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
print(new_unique_files)
if new_unique_files:
    print("new files")
    ingestion.merge_multiple_dataframe()

print("done")



##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.path.join(model_path, 'latestscore.txt'), 'rb') as f:
    old_f1 = f.readline().decode('utf8').split(' ')[1]

print(old_f1)

newest_data = os.path.join(output_folder_path, 'finaldata.csv')
new_f1 = scoring.score_model(newest_data)
print("new f1" + new_f1)


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if new_f1 < old_f1:
    print("model drift")
else:
    exit()

training.train_model()

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
deployment.store_model_into_pickle('trainedmodel.pkl')

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
reporting.score_model()
apicalls.test_app_calls()






