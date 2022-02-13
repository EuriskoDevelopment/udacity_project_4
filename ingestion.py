from unittest import skip
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    df_list = pd.DataFrame(columns=['corporation', 'lastmonth_activity', 'lastyear_activity', 'number_of_employees', 'exited'])
    ingested_files = ''
    
    filenames = (x for x in os.listdir(os.path.join(os.getcwd(), input_folder_path)) if x[-4:] == '.csv')

    for filename in filenames:
        df1 = pd.read_csv(os.path.join(os.getcwd(), input_folder_path, filename))
        df_list = df_list.append(df1)
        ingested_files += (filename + '\n')
    result=df_list.drop_duplicates()

    output_dir_fullname = os.path.join(os.getcwd(), output_folder_path)
    if not os.path.exists(output_dir_fullname):
        os.mkdir(output_dir_fullname)
    result.to_csv(os.path.join(output_dir_fullname, 'finaldata.csv'), index=False)

    with open(os.path.join(output_dir_fullname, 'ingestedfiles.txt'), 'w') as f:
        f.write(ingested_files)


if __name__ == '__main__':
    merge_multiple_dataframe()
