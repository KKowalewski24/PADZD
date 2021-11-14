import json
import logging
import os
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

import pandas as pd

"""
How to run:
    python classifier.py -f ../data/NYPD_Data_Preprocessed-1K.csv
    python classifier.py -f Pulpit/data/NYPD_Data_Preprocessed-1K.csv
    python E:\OneDrive\OneDrive - Politechnika Łódzka\2stopien\Semestr 2\PADZD\Program\scriptsclassifier.py -f Pulpit/data/NYPD_Data_Preprocessed-1K.csv

"""

JSON = ".json"
CSV = ".csv"
RESULTS_DIR = "classifier_output/"
LOGS_FILENAME: str = "app.log"

logging.basicConfig(
    filename=LOGS_FILENAME,
    level=logging.INFO,
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} - %(message)s"
)


def main() -> None:
    args = prepare_args()
    filepath = args.filepath
    create_directory(RESULTS_DIR)

    display_and_log("Loading data...")
    df: pd.DataFrame = pd.read_csv(filepath)
    display_and_log(f"Size of loaded data: {len(df.index)}")

    decision_tree_classification(df)

    display_and_log("Saving data to file")

    display_finish()


def decision_tree_classification(data_set: pd.DataFrame) -> None:
    X_train, X_test, y_train, y_test = data_set.spl

    min_samples_leaf_range = [
       10, 100, 1000, 10000
    ]
    max_depth_range = [100, 1000, 10000, 100000]
    n_estimators_range = [100, 1000, 10000]


    train_acc_history = []
    test_acc_history = []
    for n_estimators in n_estimators_range:
        forest = RandomForestClassifier(random_state=47,
                                        n_jobs=-1,
                                        n_estimators=n_estimators,
                                        max_depth=100,
                                        min_samples_leaf=10
                                        )
        forest.fit(X_train, y_train)
        train_acc_history.append(forest.score(X_train, y_train))
        test_acc_history.append(forest.score(X_test, y_test))
        print("n_estimators:", n_estimators, "\ttrain_acc:",
              train_acc_history[-1], "\ttest_acc:", test_acc_history[-1])
    # _plot_accuracy(413, train_acc_history, test_acc_history, "n_estimators",
    #                n_estimators_range)
    best_acc = np.max(test_acc_history)

    print("best accuracy:", best_acc)
    plt.show()


def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-f", "--filepath", required=True, type=str, help="Filepath to CSV file"
    )

    return arg_parser.parse_args()


def create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def check_types() -> None:
    subprocess.call(["mypy", "."])


def check_if_exists_in_args(arg: str) -> bool:
    return arg in sys.argv


def display_finish() -> None:
    display_and_log("------------------------------------------------------------------------")
    display_and_log("FINISHED")
    display_and_log("------------------------------------------------------------------------")


def display_and_log(text: str) -> None:
    print(text)
    logging.info(text)