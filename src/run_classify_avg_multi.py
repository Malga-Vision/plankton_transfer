def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


import argparse
import os
from pathlib import Path
import pandas as pd
import torch
from os.path import join, isfile
from os import listdir

from utils import load_config
from sklearn.linear_model import LogisticRegression
from typing import Optional, List
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np


OUTPUT_FOLDER = "outputs/features/"
OUTPUT_PATH = "outputs/"
GRID_PARAMS = load_config("config/grid_search_classifiers.yaml")

parser = argparse.ArgumentParser(description='Train a classifier Parser')
parser.add_argument('--dir',type=str)



def estimator_by_name(name: str):
    if name.lower() == "svc":
        return SVC
    elif name.lower() == "logisticregression":
        return LogisticRegression


def save_results(log_path, log_name, results_df):
    """
        Save the results in a csv file. If the file already exists the results are 
        appended to the file (as new rows).

        Args:
            log_path (string): the path to the log directory.
            log_name(string): the name of the log file.
            results_dict (pd.DataFrame): the results in a pd.DataFrame
    """

    # create directory if it does not exist.
    Path(log_path).mkdir(parents=True, exist_ok=True)

    if not log_name.endswith(".csv"):
        log_name += ".csv"

    log_full_path = os.path.join(log_path, log_name)

    # try append row
    try:
        old = pd.read_csv(log_full_path)
        new = pd.concat([old, results_df])
    except:
        new = results_df
    
    # save new dataframe
    new.to_csv(log_full_path, encoding='utf-8', index=False)


def main(dir: str):

    train_features = [f for f in listdir(dir) if "features" in f and "train" in f]
    #test_features = [f for f in listdir(dir) if "features" in f and "test" in f]

    probabilities = None
    best_accuracy = 0.

    for i in range(len(train_features)):
        
        X_train_name = train_features[i]
        Y_train_name = X_train_name.replace("features", "labels")
        X_test_name = X_train_name.replace("train", "test")
        Y_test_name = Y_train_name.replace("train", "test")

        
        X_train_path = join(dir, X_train_name)
        Y_train_path = join(dir, Y_train_name)
        X_test_path  = join(dir, X_test_name)
        Y_test_path  = join(dir, Y_test_name)

        X_train = torch.load(X_train_path)
        Y_train = torch.load(Y_train_path)
        X_test  = torch.load(X_test_path)
        Y_test  = torch.load(Y_test_path)

        log_reg = LogisticRegression(C=1, max_iter=1000).fit(X_train, Y_train)
        current_probabilities = log_reg.predict_proba(X_test)

        if probabilities is not None:
            probabilities += current_probabilities
        else:
            probabilities = current_probabilities

        prediction = np.argmax(probabilities, axis=1)

        accuracy = accuracy_score(Y_test, prediction)

        best_accuracy = accuracy

        print(f"ACCURACY: {best_accuracy}")
        print()
        print()







if __name__=='__main__':
    args = vars(parser.parse_args())
    main(**args)
