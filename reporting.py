import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from common import load_data, numeric_predictors



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
output_model_path = os.path.join(config['output_model_path']) 



##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    X, y = load_data(os.path.join(os.getcwd(), test_data_path, 'testdata.csv'))
    
    with open( os.path.join(os.getcwd(), prod_deployment_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X)

    cfm = metrics.confusion_matrix(y, y_pred)
    classes = ["False", "True"]
    df_cfm = pd.DataFrame(cfm/np.sum(cfm), index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, fmt='.2%', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.show()
    cfm_plot.figure.savefig(os.path.join(os.getcwd(), output_model_path, 'confusionmatrix.png'))



if __name__ == '__main__':
    score_model()
