from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from sklearn.cluster import KMeans
from module.Logger import Logger
from module.utils import display_finish
from kmodes.kmodes import KModes
from datetime import datetime
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.cm as cm
import prince
import itertools
pd.options.mode.chained_assignment = None  # default='warn'


def main() -> None:
    logger = Logger().get_logging_instance()
    args = prepare_args()
    logger.info("Start clustering with args: " + str(vars(args)))
    save_data = args.save

    logger.info("Loading dataset...")
    df = pd.read_csv("../data/NYPD_Data_Preprocessed_v2-225533.csv", nrows=1000000)
    logger.info("Loading dataset finished")
    process_clustering(df)
    display_finish()


def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-s", "--save", default=False, action="store_true", help="Save generated data"
    )

    return arg_parser.parse_args()


CRIME_LABELS = ['VIOLATION', 'MISDEMEANOR', 'FELONY']


# region Encoding
def encode_crime_label(label):
    if label == 'FELONY':
        return 2
    if label == 'MISDEMEANOR':
        return 1
    if label == 'VIOLATION':
        return 0
    else:
        return


def encode_occurance_time(label):
    date_time_obj = datetime.strptime(label, '%H:%M:%S').hour
    if 1 <= date_time_obj < 5:
        return 'BEFORE_DAWN'
    if 5 <= date_time_obj < 12:
        return 'MORNING'
    if 12 <= date_time_obj < 17:
        return 'AFTERNOON'
    if 17 <= date_time_obj < 20:
        return 'EVENING'
    if 20 <= date_time_obj <= 23:
        return 'NIGHT'
    if date_time_obj == 0:
        return 'MIDNIGHT'
    else:
        return 'UNKNOWN'
# endregion


def preprocess_initial_dataframe(df: pd.DataFrame):
    data_to_access: pd.DataFrame = df.loc[:, ['LAW_CAT_CD', 'CMPLNT_FR_TM', 'SUSP_RACE', 'SUSP_SEX',
                                              'SUSP_AGE_GROUP', 'VIC_RACE', 'VIC_AGE_GROUP', 'VIC_SEX']].copy()
    data_to_access.dropna(inplace=True)
    data_to_access['LAW_CAT_CD'] = data_to_access['LAW_CAT_CD'].apply(lambda x: encode_crime_label(x))
    data_to_access['CMPLNT_FR_TM'] = data_to_access['CMPLNT_FR_TM'].apply(lambda x: encode_occurance_time(x))
    return data_to_access


def plot_silhouette_score(cluster_labels, cluster_centers, values, n_clusters):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(values) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(values, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(values, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        values[:, 0], values[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = cluster_centers

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for results clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )


def predict_outcome(actual_labels, results):
    unique_labels_from_clustering = np.unique(results)
    unique_labels = np.unique(actual_labels)

    permutations = [list(zip(x, unique_labels_from_clustering)) for x in
                    itertools.permutations(unique_labels, len(unique_labels_from_clustering))]
    accuracies = []
    for p in permutations:
        accurate = 0
        for l in p:
            for idx, record in enumerate(actual_labels):
                if record == l[0] and results[idx] == l[1]:
                    accurate += 1
        accuracies.append(accurate / len(actual_labels) * 100.0)
    index_of_highest_permutation = np.where(accuracies == np.amax(accuracies))[0][0]
    optimal_permutation = permutations[index_of_highest_permutation]

    converted_results = ""
    for index, l in enumerate(optimal_permutation):
        converted_results += str(CRIME_LABELS[optimal_permutation[index][0]]) + " - label: " + str(optimal_permutation[index][1]) + "\n"

    print(
        "Top accuracy for permutation: \n" + str(converted_results) + str(
            np.amax(accuracies)))


def one_hot_experiment(df: pd.DataFrame):
    columns_to_one_hot = ['SUSP_RACE', 'SUSP_SEX', 'SUSP_AGE_GROUP', 'VIC_RACE', 'VIC_AGE_GROUP', 'VIC_SEX', 'CMPLNT_FR_TM']
    for column in columns_to_one_hot:
        new_one_hot_columns = pd.get_dummies(df[column])
        df = pd.concat([df, new_one_hot_columns], axis=1)
        df = df.drop(column, axis=1)
    actual_labels = np.array(df['LAW_CAT_CD'])
    data_with_removed_label = df.drop('LAW_CAT_CD', axis=1)
    array_to_process = np.array(data_with_removed_label)

    range_n_clusters = [3]

    for n_clusters in range_n_clusters:
        cluster_labels, cluster_centers, values = find_important_weights(array_to_process, n_clusters, KMeans)
        plot_silhouette_score(cluster_labels, cluster_centers, values, n_clusters)

        predict_outcome(actual_labels, cluster_labels)

    plt.show()


def kmode_experiment(df: pd.DataFrame):
    array_to_process = np.array(df)
    actual_labels = np.array(df['LAW_CAT_CD'])
    range_n_clusters = [3]

    for n_clusters in range_n_clusters:
        cluster_labels, cluster_centers, values = kmode_mca(array_to_process, n_clusters)
        plot_silhouette_score(cluster_labels, cluster_centers, values, n_clusters)

        predict_outcome(actual_labels, cluster_labels)

    plt.show()


def kmode_mca(array_to_process, n_clusters):
    results = KModes(n_clusters=n_clusters, random_state=10).fit(array_to_process)
    mca = prince.MCA().fit(array_to_process)
    Xred = np.array(mca.transform(array_to_process))

    plt.figure()
    ax = plt.gca()
    scatterHs = []
    clr = ['r', 'b', 'g']

    for cluster in set(results.labels_):
        scatterHs.append(ax.scatter(Xred[results.labels_ == cluster, 0], Xred[results.labels_ == cluster, 1],
                                    color=clr[cluster], label='Cluster {}'.format(cluster)))
    plt.legend(handles=scatterHs, loc=4)
    plt.setp(ax, title='KMode clustering after MCA transformation')

    plt.show()
    return results.labels_, results.cluster_centroids_, Xred


def process_clustering(df: pd.DataFrame) -> None:
    logger = Logger().get_logging_instance()
    logger.info("Starting clustering...")
    temp_df: pd.DataFrame = preprocess_initial_dataframe(df)
    one_hot_experiment(temp_df)
    kmode_experiment(temp_df)


def find_important_weights(array, clusters_nr, clusterer):
    results = clusterer(n_clusters=clusters_nr, random_state=10).fit(array)

    # cluster with 3 random initial clusters
    # PCA on orig. dataset
    # Xred will have only 2 columns, the first two princ. comps.
    # evals has shape (4,) and evecs (4,4). We need all eigenvalues
    # to determine the portion of variance
    Xred, evals, evecs = dim_red_pca(array, 2)

    xlab = '1. PC - ExpVar = {:.2f} %'.format(evals[0] / sum(evals) * 100)  # determine variance portion
    ylab = '2. PC - ExpVar = {:.2f} %'.format(evals[1] / sum(evals) * 100)
    # plot the clusters, each set separately
    plt.figure()
    ax = plt.gca()
    scatterHs = []
    clr = ['r', 'b', 'g']
    for cluster in set(results.labels_):
        scatterHs.append(ax.scatter(Xred[results.labels_ == cluster, 0], Xred[results.labels_ == cluster, 1],
                                    color=clr[cluster], label='Cluster {}'.format(cluster)))
    plt.legend(handles=scatterHs, loc=4)
    plt.setp(ax, title='First and Second Principle Components', xlabel=xlab, ylabel=ylab)
    # plot also the eigenvectors for deriving the influence of each feature
    fig, ax = plt.subplots(2, 1)
    indices = np.arange(0, len(evecs[0]))
    ax[0].bar(indices, evecs[0])
    plt.setp(ax[0], title="First and Second Component's Eigenvectors ", ylabel='Weight')
    ax[1].bar(indices, evecs[1])
    plt.setp(ax[1], xlabel='Features', ylabel='Weight')
    plt.show()
    return results.labels_, results.cluster_centers_, Xred


def dim_red_pca(X, d=0, corr=False):
    r"""
    Performs principal component analysis.

    Parameters
    ----------
    X : array, (n, d)
        Original observations (n observations, d features)

    d : int
        Number of principal components (default is ``0`` => all components).

    corr : bool
        If true, the PCA is performed based on the correlation matrix.

    Notes
    -----
    Always all eigenvalues and eigenvectors are returned,
    independently of the desired number of components ``d``.

    Returns
    -------
    Xred : array, (n, m or d)
        Reduced data matrix

    e_values : array, (m)
        The eigenvalues, sorted in descending manner.

    e_vectors : array, (n, m)
        The eigenvectors, sorted corresponding to eigenvalues.

    """
    # Center to average
    X_ = X - X.mean(0)
    # Compute correlation / covarianz matrix
    if corr:
        CO = np.corrcoef(X_.T)
    else:
        CO = np.cov(X_.T)
    # Compute eigenvalues and eigenvectors
    e_values, e_vectors = sp.linalg.eigh(CO)

    # Sort the eigenvalues and the eigenvectors descending
    idx = np.argsort(e_values)[::-1]
    e_vectors = e_vectors[:, idx]
    e_values = e_values[idx]
    # Get the number of desired dimensions
    d_e_vecs = e_vectors
    if d > 0:
        d_e_vecs = e_vectors[:, :d]
    else:
        d = None
    # Map principal components to original data
    LIN = np.dot(d_e_vecs, np.dot(d_e_vecs.T, X_.T)).T
    return LIN[:, :d], e_values, e_vectors


if __name__ == '__main__':
    main()
