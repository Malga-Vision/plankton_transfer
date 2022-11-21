def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


import argparse
import os
from pathlib import Path
import pandas as pd
import torch
from os.path import join
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
parser.add_argument('--dataset_train',type=str)
parser.add_argument('--dataset_test', type=str)
parser.add_argument('--model_names', nargs="+", type=str)


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


def main(model_names: List[str], dataset_train: str, dataset_test: Optional[str] = None):
    """
        Args:
            model_name (str): the name of the feature extractor model.
            dataset_train (str): the name of the training dataset.
            dataset_test (str, optional): the name of the test dataset.

        NOTE: if the dataset_test is None, the dataset_train will be split into 
        train (80%) and test (20%).

    """

    
    probabilities = None
    best_accuracy = 0
    best_models = []

    for i in range(len(model_names)):

        current_model = model_names[i]
        print(f"New model: {current_model}")

        X_train_path = join(OUTPUT_FOLDER,  f"{current_model}_{dataset_train}_features")
        Y_train_path = join(OUTPUT_FOLDER,  f"{current_model}_{dataset_train}_labels")
        X_train = torch.load(X_train_path)
        Y_train = torch.load(Y_train_path)

        if dataset_test is not None and dataset_test.lower() != "none":
            X_test_path  = join(OUTPUT_FOLDER,  f"{current_model}_{dataset_test}_features")
            Y_test_path  = join(OUTPUT_FOLDER,  f"{current_model}_{dataset_test}_labels")
            X_test  = torch.load(X_test_path)
            Y_test  = torch.load(Y_test_path)
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, 
                                                                test_size=0.20, 
                                                                random_state=42)

        #scaler = preprocessing.StandardScaler().fit(X_train)
        #X_train = scaler.transform(X_train)
        #X_test = scaler.transform(X_test)

        pca = PCA(n_components=500, whiten=True)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        log_reg = LogisticRegression(C=1, max_iter=1000).fit(X_train, Y_train)
        current_probabilities = log_reg.predict_proba(X_test)

        if probabilities is not None:
            probabilities += current_probabilities
        else:
            probabilities = current_probabilities

        prediction = np.argmax(probabilities, axis=1)

        accuracy = accuracy_score(Y_test, prediction)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_models.append(current_model)
        else:
            probabilities -= current_probabilities


        print(f"MODELS: {best_models}")
        print(f"ACCURACY: {best_accuracy}")
        print()
        print()







if __name__=='__main__':
    args = vars(parser.parse_args())
    main(**args)
