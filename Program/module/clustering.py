import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
import numpy as np
import itertools
from module.Logger import Logger

pd.options.mode.chained_assignment = None  # default='warn'

experiments = {
    'single': ['SUSP_SEX', 'VIC_RACE'],
    'multi': [['SUSP_SEX', 'VIC_SEX'], ['SUSP_RACE', 'VIC_RACE']]
}


def process_clustering(df: pd.DataFrame) -> None:
    logger = Logger().get_logging_instance()
    logger.info("Starting clustering...")
    for type in experiments.keys():
        print("Experiments for label type: " + type)
        if type == 'single':
            for column_name in experiments[type]:
                calculate_single_label_data(column_name, df)
        else:
            for columns in experiments[type]:
                calculate_multilabel_data(columns, df)
    logger.info("Clustering finished...")


def initial_processing_for_single_label(dataset: pd.DataFrame, label) -> pd.DataFrame:
    dataset = dataset.dropna()
    dataset = dataset.query(f'{label} != "OTHER"')
    cropped_data = pd.DataFrame(dataset.query(f'{label} != "OTHER"')).reset_index()
    actual_labels = dataset[label]
    print("Number of analised records for label: " + label + " " + str(len(dataset)))
    dataset.drop(columns=[label], inplace=True)
    dataset = label_encode_columns(dataset)

    return dataset, actual_labels, cropped_data


def initial_processing_for_multi_label(dataset: pd.DataFrame, labels) -> pd.DataFrame:
    dataset = dataset.dropna()

    mixed_label = create_mixed_label(labels)

    for label in labels:
        dataset = dataset.query(f'{label} != "OTHER"')

    dataset[mixed_label] = dataset[labels[0]] + "_" + dataset[labels[1]]
    actual_labels = dataset[mixed_label]

    cropped_data = pd.DataFrame(dataset).reset_index()

    for label in labels:
        dataset.drop(columns=[label], inplace=True)
    print("Number of analised records for label: " + mixed_label + " " + str(len(dataset)))
    dataset = label_encode_columns(dataset)
    return dataset, actual_labels, cropped_data


def label_encode_columns(dataset):
    labels_to_transform = []

    for r in range(dataset.shape[1]):
        column_value = dataset.iloc[0, r]
        if isinstance(column_value, str):
            labels_to_transform.append(r)
    dataset.iloc[:, labels_to_transform] = LabelEncoder().fit_transform(dataset.iloc[0, labels_to_transform])
    return dataset


def calculate_multilabel_data(columns, df):
    print("Mixed labels: " + str(columns))

    df_to_process, actual_labels, cropped_data = initial_processing_for_multi_label(df, columns)
    actual_labels = pd.DataFrame(actual_labels)
    column_name = create_mixed_label(columns)

    clasterize(column_name, actual_labels, df_to_process, cropped_data)


def calculate_single_label_data(column_name, df):
    print("\nLabel: " + column_name)
    df_to_process, actual_labels, cropped_data = initial_processing_for_single_label(df, column_name)
    clasterize(column_name, actual_labels, df_to_process, cropped_data)


def clasterize(column_name, actual_labels, df_to_process, cropped_data):
    actual_labels = pd.DataFrame(actual_labels)
    unique_actual_labels = actual_labels[column_name].unique()
    print("--------------------------------------------------------------------")
    print("KMeans")
    k_means = KMeans(
        n_clusters=len(unique_actual_labels)
    )
    labels_k_means = k_means.fit_predict(df_to_process)
    get_accuracy(cropped_data, labels_k_means, unique_actual_labels, column_name)

    print("\nAgglomerative")
    agglomerative = AgglomerativeClustering(linkage="single", n_clusters=len(unique_actual_labels))
    labels_k_agglomerative = agglomerative.fit_predict(df_to_process)
    get_accuracy(cropped_data, labels_k_agglomerative, unique_actual_labels, column_name)

    print("\nDBSCAN")
    dbscan = DBSCAN()
    dbscan_labels = dbscan.fit_predict(df_to_process)
    print(dbscan_labels)
    print("DBSCAN created: " + str(len(np.unique(dbscan_labels))) + " number of clusters")


def get_accuracy(cropped_data, labels_from_clustering, unique_actual_labels, column_name):
    unique_labels_from_clustering = np.unique(labels_from_clustering)
    permutations = [list(zip(x, unique_labels_from_clustering)) for x in
                    itertools.permutations(unique_actual_labels, len(unique_labels_from_clustering))]
    accuracies = []
    for p in permutations:
        accurate = 0
        for l in p:
            for (idx, record) in cropped_data.iterrows():
                if record[column_name] == l[0] and labels_from_clustering[idx] == l[1]:
                    accurate += 1
        accuracies.append(accurate / len(cropped_data) * 100.0)
    index_of_highest_permutation = np.where(accuracies == np.amax(accuracies))[0][0]
    print(
        "Top accuracy for permutation " + str(permutations[index_of_highest_permutation]) + ": " + str(
            np.amax(accuracies)))


def create_mixed_label(columns):
    return columns[0] + "_" + columns[1]
