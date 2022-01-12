import logging
import os
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics, naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from shared.LatexGenerator import LatexGenerator
from shared.label_names_mapper import *
from shared.utils import display_finish

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

LATEX_RESULTS_DIR = "latex_results"
latex_generator: LatexGenerator = LatexGenerator(LATEX_RESULTS_DIR)


def main() -> None:
    # args = prepare_args()
    # filepath = args.filepath
    filepath = "../data/NYPD_Data_Preprocessed_v2.csv"
    create_directory(RESULTS_DIR)

    display_and_log("Loading data...")
    df: pd.DataFrame = pd.read_csv(filepath, nrows=1000000)
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
    # decision_tree_classification(data_set=df,
    #                                        label_to_classifier=LawBreakingLabels.LAW_BREAKING_LEVEL,
    #                                        test_percentage=0.3)
    decision_tree_classification_final(data_set=df,
                                       # label_to_classifier=LawBreakingLabels.LAW_BREAKING_LEVEL,
                                       label_to_classifier=LawBreakingLabels.KEY_CODE,
                                       test_percentage=0.3,
                                       estimators=1000,
                                       leaf=100,
                                       samples=0.99,
                                       depth=15)

    # bayes_classification(df, 0.3)

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


def decision_tree_classification(data_set: pd.DataFrame, label_to_classifier:str,
                                 test_percentage: float) -> None:
    print("Test data percentage: " + str(round(test_percentage * 100, 2)) + "%")

    train, test = train_test_split(data_set, test_size=test_percentage)
    #
    # y_train = train[LawBreakingLabels.KEY_CODE]
    # x_train = train.drop(columns=[LawBreakingLabels.KEY_CODE])
    # y_test = test[LawBreakingLabels.KEY_CODE]
    # x_test = test.drop(columns=[LawBreakingLabels.KEY_CODE])

    y_train = train[label_to_classifier]
    x_train = train.drop(columns=[label_to_classifier])
    y_test = test[label_to_classifier]
    x_test = test.drop(columns=[label_to_classifier])

    min_samples_leaf_range = [
       10, 100, 1000, 10000
    ]
    max_depth_range = [5, 10, 15, 25]
    n_estimators_range = [100, 1000]
    max_samples_range = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.99]

########################################################################################################################
### min_samples_leaf_range
    train_acc_history = []
    test_acc_history = []
    for min_samples_leaf in min_samples_leaf_range:
        tree = DecisionTreeClassifier(random_state=47,
                                      min_samples_leaf=min_samples_leaf)
        tree.fit(x_train, y_train)
        train_acc_history.append(tree.score(x_train, y_train))
        test_acc_history.append(tree.score(x_test, y_test))
        print("min_samples_leaf:", min_samples_leaf, "\ttrain_acc:",
              train_acc_history[-1], "\ttest_acc:", test_acc_history[-1])
    best_acc = np.max(test_acc_history)
    best_params = {
        "min_samples_leaf": min_samples_leaf_range[np.argmax(test_acc_history)]
    }

########################################################################################################################
### max_depth_range
    train_acc_history = []
    test_acc_history = []
    for max_depth in max_depth_range:
        tree = DecisionTreeClassifier(random_state=47, max_depth=max_depth)
        tree.fit(x_train, y_train)
        train_acc_history.append(tree.score(x_train, y_train))
        test_acc_history.append(tree.score(x_test, y_test))
        print("max_depth:", max_depth, "\ttrain_acc:", train_acc_history[-1],
              "\ttest_acc:", test_acc_history[-1])

    if best_acc < np.max(test_acc_history):
        best_acc = np.max(test_acc_history)
        best_params = {
            "max_depth": max_depth_range[np.argmax(test_acc_history)]
        }
    print("best params for single tree:", best_params)

########################################################################################################################
### n_estimators_range
    train_acc_history = []
    test_acc_history = []
    for n_estimators in n_estimators_range:
        forest = RandomForestClassifier(random_state=47,
                                        n_jobs=-1,
                                        n_estimators=n_estimators,
                                        **best_params)
        forest.fit(x_train, y_train)
        train_acc_history.append(forest.score(x_train, y_train))
        test_acc_history.append(forest.score(x_test, y_test))
        print("n_estimators:", n_estimators, "\ttrain_acc:",
              train_acc_history[-1], "\ttest_acc:", test_acc_history[-1])
    best_acc = np.max(test_acc_history)
    best_params['n_estimators'] = n_estimators_range[np.argmax(
        test_acc_history)]
########################################################################################################################
### max_samples_range
    train_acc_history = []
    test_acc_history = []
    for max_samples in max_samples_range:
        forest = RandomForestClassifier(random_state=47,
                                        n_jobs=-1,
                                        max_samples=max_samples,
                                        **best_params)
        forest.fit(x_train, y_train)
        train_acc_history.append(forest.score(x_train, y_train))
        test_acc_history.append(forest.score(x_test, y_test))
        print("max_samples:", max_samples, "\ttrain_acc:",
              train_acc_history[-1], "\ttest_acc:", test_acc_history[-1])

    best_acc = np.max(test_acc_history)
    best_params['max_samples'] = max_samples_range[np.argmax(test_acc_history)]

    print("best params:", best_params, "best accuracy:", best_acc)


########################################################################################################################
### Final test
def decision_tree_classification_final(data_set: pd.DataFrame, label_to_classifier:str,
                                       test_percentage: float, estimators:int, leaf:int, samples:float, depth:int) -> None:
    train, test = train_test_split(data_set, test_size=test_percentage)

    y_train = train[label_to_classifier]
    x_train = train.drop(columns=[label_to_classifier])
    y_test = test[label_to_classifier]
    x_test = test.drop(columns=[label_to_classifier])
    train_acc_history = []
    test_acc_history = []
    forest = RandomForestClassifier(random_state=47,
                                    n_jobs=-1,
                                    n_estimators=estimators,
                                    min_samples_leaf=leaf,
                                    max_samples=samples,
                                    max_depth=depth)
    forest.fit(x_train, y_train)
    train_acc_history.append(forest.score(x_train, y_train))
    test_acc_history.append(forest.score(x_test, y_test))
    print("\ttrain_acc:",
          train_acc_history[-1], "\ttest_acc:", test_acc_history[-1])



    y_pred = forest.predict(x_test)
    y_proba = forest.predict_proba(x_test)

    # u, indices = np.unique(y_test, return_index=True)
    # x = [indices[i] for i in (0, 16, 32, 48)]
    # x = [indices[i] for i in (0, 3, 5,? 4)]
    print("before results")

    results = {
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "accuracy": np.round(accuracy_score(y_test, y_pred), 4),
        "recall": np.round(recall_score(y_test, y_pred, average=None), 4),
        "precision": np.round(precision_score(y_test, y_pred, average=None), 4),
        # "roc_curves": [roc_curve(y_test, y_proba[:, int(index)], pos_label=int(value)) for index, value in enumerate(np.unique(y_test))],
        # "roc_curves": [roc_curve(y_test, y_proba[:, index], pos_label=u[x]) for index in x],
        # "roc_curves": rocs[0],
        # "learning_curve": learning_curve(forest, x_train, y_train, n_jobs=-1,
        #                                  train_sizes=np.linspace(0.1, 1.0, 10))
    }
    print("after results")

    # ax = plt.gca()
    # rfc_disp = RocCurveDisplay.from_estimator(forest, x_test, y_test, ax=ax, alpha=0.8)
    # plt.show()

    metrics = {"forest":results}
    save_metrics(metrics, "forest")




def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = drop_unused_columns(data)
    data = remove_na(data)
    data = extract_hour_and_day(data)

    for column in ['day_of_week_sin','day_of_week_cos','day_of_year_sin','day_of_year_cos']:
        data = data[data[column].notna()]

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
                              DateTimeEventLabels.EVENT_END_TIMESTAMP,
                              "CMPLNT_FR_TM", "CMPLNT_TO_TM", "OFNS_DESC",
                            'PD_DESC', 'JURIS_DESC', 'JURISDICTION_CODE',
                              'PARKS_NM', 'HADEVELOPT', 'HOUSING_PSA', 'X_COORD_CD', 'Y_COORD_CD', 'TRANSIT_DISTRICT',
                              'Lat_Lon', 'PATROL_BORO', 'STATION_NAME'
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
    one_hot_columns = [
        EventStatusLabels.EVENT_STATUS,
        VictimLabels.VICTIM_RACE,
        VictimLabels.VICTIM_SEX,
        SuspectLabels.SUSPECT_RACE,
        SuspectLabels.SUSPECT_SEX
    ]
    ordinal_columns: List[Tuple[str, List]] = [
        (LawBreakingLabels.LAW_BREAKING_LEVEL, ["VIOLATION", "MISDEMEANOR", "FELONY"]),
        (SuspectLabels.SUSPECT_AGE_GROUP, ["<18", "18-24", "25-44", "45-64", "65+", "UNKNOWN"]),
        (VictimLabels.VICTIM_AGE_GROUP, ["<18", "18-24", "25-44", "45-64", "65+", "UNKNOWN"]),
    ]
    rest_columns = [
        # LawBreakingLabels.KEY_CODE,
        EventSurroundingsLabels.PLACE_TYPE,
        EventSurroundingsLabels.PLACE_TYPE_POSITION
    ]

    # for column in one_hot_columns:
    #     data[column] = OneHotEncoder.fit_transform(data[column])

    display_and_log("Encoding one hot columns")
    data = pd.get_dummies(data, columns=one_hot_columns, prefix=one_hot_columns)

    for ordinal_column in ordinal_columns:
        label, categories = ordinal_column
        data[label] = OrdinalEncoder(categories=[categories]).fit_transform(data[[label]])

    # for column in rest_columns:
        # data[column] = LabelEncoder.fit_transform(data[column])

    encoder = LabelEncoder()
    for column in rest_columns:
        data[column] = encoder.fit_transform(data[column])
    return data


def transform_date_and_time(data: pd.DataFrame, days_and_hours_sin_cos: pd.DataFrame) -> pd.DataFrame:
    data['day_of_week_sin'] = np.sin(days_and_hours_sin_cos['day_of_week'] * (2 * np.pi / 7))
    data['day_of_week_cos'] = np.cos(days_and_hours_sin_cos['day_of_week'] * (2 * np.pi / 7))

    # data['hour_of_day_sin'] = np.sin(days_and_hours_sin_cos['hour_of_day'] * (2 * np.pi / 24))
    # data['hour_of_day_cos'] = np.cos(days_and_hours_sin_cos['hour_of_day'] * (2 * np.pi / 24))

    data['day_of_year_sin'] = np.sin(days_and_hours_sin_cos['day_of_year'] * (2 * np.pi / 365))
    data['day_of_year_cos'] = np.cos(days_and_hours_sin_cos['day_of_year'] * (2 * np.pi / 365))
    return data


def extract_hour_and_day(data: pd.DataFrame) -> pd.DataFrame:
    temp_df = pd.DataFrame()
    temp_df['day_of_week'] = pd.to_datetime(data[DateTimeEventLabels.EVENT_START_TIMESTAMP],  errors = 'coerce')\
        .map(lambda date: date.weekday())
    # temp_df['hour_of_day'] = pd.to_datetime(data[DateTimeEventLabels.EVENT_START_TIMESTAMP])\
    #     .map(lambda date: date.dt.hour)
    # temp_df['hour_of_day'] = pd.to_datetime(data[DateTimeEventLabels.EVENT_START_TIMESTAMP]).dt.hour
    temp_df['day_of_year'] = pd.to_datetime(data[DateTimeEventLabels.EVENT_START_TIMESTAMP],  errors = 'coerce').dt.dayofyear
    # temp_df['day_of_year'] = data[DateTimeEventLabels.EVENT_START_TIMESTAMP].dt.dayofyear
    # print(data[DateTimeEventLabels.EVENT_START_TIMESTAMP].to_string())


    for column in ['day_of_week','day_of_year']:
        temp_df = temp_df[temp_df[column].notna()]

    return transform_date_and_time(data.drop(columns=[DateTimeEventLabels.EVENT_START_TIMESTAMP]), temp_df)


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



def save_metrics(metrics, filename_prefix):
    # save tables with confusion matrices
    for classifier in metrics:
        matrix = metrics[classifier]["confusion_matrix"]
        latex_generator.generate_vertical_table(
            matrix[0], matrix[1:], filename_prefix + "_" + classifier + "_confusion_matrix"
        )

    # save tables with basic metrics
    if len(list(metrics.values())[0]["recall"]) == 2:
        matrix = [
            [classifier,
             metrics[classifier]["accuracy"],
             metrics[classifier]["recall"][1],
             metrics[classifier]["recall"][0],
             metrics[classifier]["precision"][1]]
            for classifier in metrics]
        latex_generator.generate_vertical_table(
            ["Classifier", "Accuracy", "Sensitivity", "Specificity", "Precision"],
            matrix, filename_prefix + "_basic_metrics"
        )
    else:
        matrix = [
            [classifier,
             metrics[classifier]["accuracy"],
             str(metrics[classifier]["recall"]),
             str(metrics[classifier]["precision"])]
            for classifier in metrics]
        latex_generator.generate_vertical_table(
            ["Classifier", "Accuracy", "Sensitivities", "Precisions"],
            matrix, filename_prefix + "_basic_metrics"
        )

    # save chart with ROC curve
    # number_of_roc_curves = len(list(metrics.values())[0]["roc_curves"])
    # number_of_roc_curves = 4
    # if number_of_roc_curves == 2:
    #     for classifier in metrics:
    #         fpr, tpr, _ = metrics[classifier]["roc_curves"][1]
    #         plt.plot(fpr, tpr, label=classifier)
    #     plt.legend()
    # else:
    #     for i in range(number_of_roc_curves):
    #         plt.subplot(int(np.ceil(number_of_roc_curves / 2)), 2, i + 1)
    #         plt.title("class: " + str(i))
    #         for classifier in metrics:
    #             fpr, tpr, _ = metrics[classifier]["roc_curves"][i]
    #             plt.plot(fpr, tpr, label=classifier)
    #         plt.legend()
    # plt.show()

    # # save charts with learning curves
    # for classifier, i in zip(metrics, range(len(metrics))):
    #     plt.subplot(int(np.ceil(len(metrics) / 2)), 2, i + 1)
    #     plt.title(classifier)
    #     train_sizes_abs, train_scores, test_scores, = metrics[classifier]["learning_curve"]
    #     plt.plot(train_sizes_abs, np.average(train_scores, axis=1), label="train")
    #     plt.plot(train_sizes_abs, np.average(test_scores, axis=1), label="test")
    #     plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
