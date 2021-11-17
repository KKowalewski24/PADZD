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
from sklearn import metrics, naive_bayes
from sklearn.preprocessing import LabelEncoder
import calendar

import pandas as pd

from module.label_names_mapper import *

"""
How to run:
    python classifier.py -f ../data/NYPD_Data_Preprocessed-1K.csv
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
    filepath = "../data/NYPD_Data_Preprocessed-ALL.csv"
    create_directory(RESULTS_DIR)

    display_and_log("Loading data...")
    df: pd.DataFrame = pd.read_csv(filepath, nrows=600000)
    display_and_log(f"Size of loaded data: {len(df.index)}")

    df = preprocess_data(df)

    # test_data_percentage_tree = [0.1, 0.2, 0.3, 0.35]
    # for test_percentage in test_data_percentage_tree:
    #     decision_tree_classification(df, test_percentage)
    #
    # test_data_percentage_bayes = [0.1, 0.2, 0.3, 0.35]
    # for test_percentage in test_data_percentage_bayes:
    #     bayes_classification(df, test_percentage)


    #FINAL
    decision_tree_classification(df, 0.2)
    bayes_classification(df, 0.3)

    display_finish()


def bayes_classification(data_set: pd.DataFrame, test_percentage: float) -> None:
    accuracy_list: List[List[float]] = []
    train, test = train_test_split(data_set, test_size=test_percentage)
    y_train = train[LawBreakingLabels.KEY_CODE]
    x_train = train.drop(columns=[LawBreakingLabels.KEY_CODE])
    y_test = test[LawBreakingLabels.KEY_CODE]
    x_test = test.drop(columns=[LawBreakingLabels.KEY_CODE])

    bayes_classifier = naive_bayes.GaussianNB()
    bayes_classifier.fit(x_train, y_train)
    y_prediction = bayes_classifier.predict(x_test)
    accuracy = round(metrics.accuracy_score(y_test, y_prediction), 4)
    accuracy_list.append([accuracy])
    print("Test data percentage: " + str(
        round(test_percentage * 100, 2)) + "% ,\t" + "accuracy: " + str(accuracy))


def decision_tree_classification(data_set: pd.DataFrame, test_percentage: float) -> None:
#     print("Test data percentage: " + str(round(test_percentage * 100, 2)) + "%")
#
#     train, test = train_test_split(data_set, test_size=test_percentage)
#
#     y_train = train[LawBreakingLabels.KEY_CODE]
#     x_train = train.drop(columns=[LawBreakingLabels.KEY_CODE])
#     y_test = test[LawBreakingLabels.KEY_CODE]
#     x_test = test.drop(columns=[LawBreakingLabels.KEY_CODE])
#
#     min_samples_leaf_range = [
#        10, 100, 1000, 10000
#     ]
#     max_depth_range = [100, 1000, 10000, 100000]
#     n_estimators_range = [100, 1000]
#     max_samples_range = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.99]
#
# ########################################################################################################################
# ### min_samples_leaf_range
#     train_acc_history = []
#     test_acc_history = []
#     for min_samples_leaf in min_samples_leaf_range:
#         tree = DecisionTreeClassifier(random_state=47,
#                                       min_samples_leaf=min_samples_leaf)
#         tree.fit(x_train, y_train)
#         train_acc_history.append(tree.score(x_train, y_train))
#         test_acc_history.append(tree.score(x_test, y_test))
#         print("min_samples_leaf:", min_samples_leaf, "\ttrain_acc:",
#               train_acc_history[-1], "\ttest_acc:", test_acc_history[-1])
#     best_acc = np.max(test_acc_history)
#     best_params = {
#         "min_samples_leaf": min_samples_leaf_range[np.argmax(test_acc_history)]
#     }
#
# ########################################################################################################################
# ### max_depth_range
#     train_acc_history = []
#     test_acc_history = []
#     for max_depth in max_depth_range:
#         tree = DecisionTreeClassifier(random_state=47, max_depth=max_depth)
#         tree.fit(x_train, y_train)
#         train_acc_history.append(tree.score(x_train, y_train))
#         test_acc_history.append(tree.score(x_test, y_test))
#         print("max_depth:", max_depth, "\ttrain_acc:", train_acc_history[-1],
#               "\ttest_acc:", test_acc_history[-1])
#
#     if best_acc < np.max(test_acc_history):
#         best_acc = np.max(test_acc_history)
#         best_params = {
#             "max_depth": max_depth_range[np.argmax(test_acc_history)]
#         }
#     print("best params for single tree:", best_params)
#
# ########################################################################################################################
# ### n_estimators_range
#     train_acc_history = []
#     test_acc_history = []
#     for n_estimators in n_estimators_range:
#         forest = RandomForestClassifier(random_state=47,
#                                         n_jobs=-1,
#                                         n_estimators=n_estimators,
#                                         **best_params)
#         forest.fit(x_train, y_train)
#         train_acc_history.append(forest.score(x_train, y_train))
#         test_acc_history.append(forest.score(x_test, y_test))
#         print("n_estimators:", n_estimators, "\ttrain_acc:",
#               train_acc_history[-1], "\ttest_acc:", test_acc_history[-1])
#     best_acc = np.max(test_acc_history)
#     best_params['n_estimators'] = n_estimators_range[np.argmax(
#         test_acc_history)]
# ########################################################################################################################
# ### max_samples_range
#     train_acc_history = []
#     test_acc_history = []
#     for max_samples in max_samples_range:
#         forest = RandomForestClassifier(random_state=47,
#                                         n_jobs=-1,
#                                         max_samples=max_samples,
#                                         **best_params)
#         forest.fit(x_train, y_train)
#         train_acc_history.append(forest.score(x_train, y_train))
#         test_acc_history.append(forest.score(x_test, y_test))
#         print("max_samples:", max_samples, "\ttrain_acc:",
#               train_acc_history[-1], "\ttest_acc:", test_acc_history[-1])
#
#     best_acc = np.max(test_acc_history)
#     best_params['max_samples'] = max_samples_range[np.argmax(test_acc_history)]
#
#     print("best params:", best_params, "best accuracy:", best_acc)

########################################################################################################################
### Final test
    train, test = train_test_split(data_set, test_size=test_percentage)

    y_train = train[LawBreakingLabels.KEY_CODE]
    x_train = train.drop(columns=[LawBreakingLabels.KEY_CODE])
    y_test = test[LawBreakingLabels.KEY_CODE]
    x_test = test.drop(columns=[LawBreakingLabels.KEY_CODE])

    train_acc_history = []
    test_acc_history = []
    forest = RandomForestClassifier(random_state=47,
                                    n_jobs=-1,
                                    n_estimators=1000,
                                    min_samples_leaf=100,
                                    max_samples=0.99,
                                    max_depth=100)
    forest.fit(x_train, y_train)
    train_acc_history.append(forest.score(x_train, y_train))
    test_acc_history.append(forest.score(x_test, y_test))
    print("\ttrain_acc:",
          train_acc_history[-1], "\ttest_acc:", test_acc_history[-1])


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = drop_unused_columns(data)
    data = remove_na(data)
    data = extract_hour_and_day(data)
    data = transform_labels(data)
    print("Data rows count, after preprocessing: ", data.shape[0])
    return data


def drop_unused_columns(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(columns=[IdentifierLabels.ID,
                              LawBreakingLabels.PD_CODE,
                              EventLocationLabels.PRECINCT_CODE,
                              EventLocationLabels.BOROUGH_NAME,
                              EventLocationLabels.LATITUDE,
                              EventLocationLabels.LONGITUDE,
                              # SuspectLabels.SUSPECT_AGE_GROUP,
                              # VictimLabels.VICTIM_AGE_GROUP,
                              DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_TIMESTAMP,
                              DateTimeEventLabels.EVENT_END_TIMESTAMP
                              ])


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
        SuspectLabels.SUSPECT_AGE_GROUP,
        SuspectLabels.SUSPECT_RACE,
        SuspectLabels.SUSPECT_SEX,
        VictimLabels.VICTIM_AGE_GROUP,
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
        VictimLabels.VICTIM_AGE_GROUP,
        SuspectLabels.SUSPECT_RACE,
        SuspectLabels.SUSPECT_SEX,
        SuspectLabels.SUSPECT_AGE_GROUP,
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
