import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from module.label_names_mapper import *
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from random_forest.preprocess import *


def main() -> None:
    filepath = "../data/NYPD_Data_Preprocessed_v2.csv"
    # create_directory(RESULTS_DIR)

    df: pd.DataFrame = pd.read_csv(filepath, nrows=1000)
    df = preprocess_data(df)

    decision_tree_classification(data_set=df,
                                 label_to_classifier=LawBreakingLabels.KEY_CODE,
                                 test_percentage=.2)


def decision_tree_classification(data_set: pd.DataFrame,
                                 label_to_classifier:str,
                                 test_percentage: float) -> None:
    train, test = train_test_split(data_set, test_size=test_percentage)

    y_train = train[label_to_classifier]
    x_train = train.drop(columns=[label_to_classifier])
    y_test = test[label_to_classifier]
    x_test = test.drop(columns=[label_to_classifier])
    train_acc_history = []
    test_acc_history = []
    forest = RandomForestClassifier(random_state=47,
                                    n_jobs=-1)


    forest.fit(x_train, y_train)
    train_acc_history.append(forest.score(x_train, y_train))
    test_acc_history.append(forest.score(x_test, y_test))
    print("\ttrain_acc:",
          train_acc_history[-1], "\ttest_acc:", test_acc_history[-1])

    specify_features_importances(forest=forest,
                                 x_test=x_test,
                                 y_test=y_test)


def specify_features_importances(forest: RandomForestClassifier, x_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    _mean_decrease(forest, feature_names=x_test.columns)
    _feature_permutation(forest, x_test, y_test, feature_names=x_test.columns)


def _mean_decrease(forest: RandomForestClassifier, feature_names) -> None:
    start_time = time.time()
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    forest_importances = pd.Series(importances, index=feature_names)

    _draw_plot_bar(forest_importances, std, "Feature importances using MDI", "Mean decrease in impurity")


def _feature_permutation(forest: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame, feature_names) -> None:
    start_time = time.time()
    result = permutation_importance(
        forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(result.importances_mean, index=feature_names)

    _draw_plot_bar(forest_importances, result.importances_std, "Feature importances using permutation on full model", "Mean accuracy decrease")


def _draw_plot_bar(forest_importances, yerr, title, ylabel) -> None:
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=yerr, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()