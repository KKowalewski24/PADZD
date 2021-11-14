import json
import logging
import os
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Dict, List, Tuple
from module.utils import display_finish, run_main

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import calendar

import pandas as pd

from module.label_names_mapper import *

"""
How to run:
    python classifier.py -f ../data/NYPD_Data_Preprocessed-1K.csv
    python classifier.py module/label_names_mapper.py -f ../data/NYPD_Data_Preprocessed-1K.csv
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
    # args = prepare_args()
    # filepath = args.filepath
    filepath = "../data/NYPD_Data_Preprocessed-1K.csv"
    create_directory(RESULTS_DIR)

    display_and_log("Loading data...")
    df: pd.DataFrame = pd.read_csv(filepath)
    display_and_log(f"Size of loaded data: {len(df.index)}")

    df = preprocess_data(df)
    decision_tree_classification(df)

    display_and_log("Saving data to file")

    display_finish()


def decision_tree_classification(data_set: pd.DataFrame) -> None:
    train, test = train_test_split(data_set, test_size=0.2)

    y_train = train[LawBreakingLabels.KEY_CODE]
    x_train = train.drop(columns=[LawBreakingLabels.KEY_CODE])
    y_test = test[LawBreakingLabels.KEY_CODE]
    x_test = test.drop(columns=[LawBreakingLabels.KEY_CODE])

    min_samples_leaf_range = [
       10, 100, 1000, 10000
    ]
    max_depth_range = [100, 1000, 10000, 100000]
    n_estimators_range = [100, 1000, 10000]


    forest = RandomForestClassifier(random_state=47,
                                        n_jobs=-1,
                                        n_estimators=100,
                                        max_depth=100,
                                        min_samples_leaf=10
                                        )
    forest.fit(x_train, y_train)
    train_acc_history = []
    test_acc_history = []
    train_acc_history.append(forest.score(x_train, y_train))
    test_acc_history.append(forest.score(x_test, y_test))

    print("n_estimators:", 100, "\ttrain_acc:",
          train_acc_history[-1], "\ttest_acc:", test_acc_history[-1])

    # train_acc_history = []
    # test_acc_history = []
    # for n_estimators in n_estimators_range:
    #     forest = RandomForestClassifier(random_state=47,
    #                                     n_jobs=-1,
    #                                     n_estimators=n_estimators,
    #                                     max_depth=100,
    #                                     min_samples_leaf=10
    #                                     )
    #     forest.fit(X_train, y_train)
    #     train_acc_history.append(forest.score(X_train, y_train))
    #     test_acc_history.append(forest.score(X_test, y_test))
    #     print("n_estimators:", n_estimators, "\ttrain_acc:",
    #           train_acc_history[-1], "\ttest_acc:", test_acc_history[-1])
    # # _plot_accuracy(413, train_acc_history, test_acc_history, "n_estimators",
    # #                n_estimators_range)
    # best_acc = np.max(test_acc_history)
    #
    # print("best accuracy:", best_acc)
    # plt.show()


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop(columns=[IdentifierLabels.ID,
                              LawBreakingLabels.PD_CODE,
                              EventLocationLabels.PRECINCT_CODE,
                              EventLocationLabels.BOROUGH_NAME,
                              EventLocationLabels.LATITUDE,
                              EventLocationLabels.LONGITUDE,
                              SuspectLabels.SUSPECT_AGE_GROUP,
                              VictimLabels.VICTIM_AGE_GROUP,
                              DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_TIMESTAMP,
                              DateTimeEventLabels.EVENT_END_TIMESTAMP
                              ])

    data = remove_na(data)
    data = extract_hour_and_day(data)
    data = transform_labels(data)
    return data


def remove_na(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        DateTimeEventLabels.EVENT_START_TIMESTAMP,
        # DateTimeEventLabels.EVENT_END_TIMESTAMP,
        # DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_TIMESTAMP,
        LawBreakingLabels.KEY_CODE,
        LawBreakingLabels.LAW_BREAKING_LEVEL,
        EventStatusLabels.EVENT_STATUS,
        EventSurroundingsLabels.PLACE_TYPE,
        EventSurroundingsLabels.PLACE_TYPE_POSITION,
        # SuspectLabels.SUSPECT_AGE_GROUP,
        SuspectLabels.SUSPECT_RACE,
        SuspectLabels.SUSPECT_SEX,
        # VictimLabels.VICTIM_AGE_GROUP,
        VictimLabels.VICTIM_RACE,
        VictimLabels.VICTIM_SEX
    ]
    for column in columns:
        data = data[data[column].notna()]
    return data


def transform_labels(data: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = [
        EventStatusLabels.EVENT_STATUS,
        LawBreakingLabels.LAW_BREAKING_LEVEL,
        EventSurroundingsLabels.PLACE_TYPE,
        EventSurroundingsLabels.PLACE_TYPE_POSITION,
        VictimLabels.VICTIM_RACE,
        VictimLabels.VICTIM_SEX,
        SuspectLabels.SUSPECT_RACE,
        SuspectLabels.SUSPECT_SEX,
        'DayOfWeek'
    ]
    encoder = LabelEncoder()
    for column in categorical_columns:
        data[column] = encoder.fit_transform(data[column])
    return data


def extract_hour_and_day(data: pd.DataFrame) -> pd.DataFrame:
    data['DayOfWeek'] = pd.to_datetime(data[DateTimeEventLabels.EVENT_START_TIMESTAMP])\
        .map(lambda date: calendar.day_name[date.weekday()])
    data['Hour'] = pd.to_datetime(data[DateTimeEventLabels.EVENT_START_TIMESTAMP]) \
        .map(lambda date: date.hour)
    data = data.drop(columns=[DateTimeEventLabels.EVENT_START_TIMESTAMP])
    return data

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

if __name__ == "__main__":
    main()
