import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import numpy as np

RESULTS_DIR = "saved_plots/"

def specify_features_importances(forest: RandomForestClassifier, x_test: pd.DataFrame, y_test: pd.DataFrame, processed_label: str) -> None:
    _mean_decrease(forest, feature_names=x_test.columns, processed_label=processed_label)
    _feature_permutation(forest, x_test, y_test, feature_names=x_test.columns, processed_label=processed_label)


def _mean_decrease(forest: RandomForestClassifier, feature_names, processed_label: str) -> None:
    start_time = time.time()
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances (mean_decrease): {elapsed_time:.3f} seconds")
    forest_importances = pd.Series(importances, index=feature_names)

    _draw_plot_bar(forest_importances, std, "Feature importances using MDI", "Mean decrease in impurity", processed_label)


def _feature_permutation(forest: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame, feature_names, processed_label: str) -> None:
    start_time = time.time()
    result = permutation_importance(
        forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances (feature_permutation): {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(result.importances_mean, index=feature_names)

    _draw_plot_bar(forest_importances, result.importances_std,
                   "Feature importances using permutation on full model",
                   "Mean accuracy decrease", processed_label
                   )


def _draw_plot_bar(forest_importances: pd.Series, yerr, title, xlabel, processed_label: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    forest_importances.plot.barh(color='green', xerr=yerr, ax=ax)
    ax.set_title(title + " for " + processed_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Feature name")

    plt.savefig(RESULTS_DIR + title + " for " + processed_label, bbox_inches="tight")
    plt.close()
    plt.show()
