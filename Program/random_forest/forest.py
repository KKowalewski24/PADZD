from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from random_forest.preprocess import *
from random_forest.features_importance import specify_features_importances
import matplotlib.pyplot as plt


def main() -> None:
    filepath = "../data/NYPD_Data_Preprocessed_v2.csv"
    # create_directory(RESULTS_DIR)

    df: pd.DataFrame = pd.read_csv(filepath, nrows=1000)
    df = preprocess_data(df)
    df_copy = df.copy()
    process_key_code(df)
    process_law_breaking_level(df_copy)


def process_key_code(data_set: pd.DataFrame) -> None:
    data_set = preprocess_data_to_classifer(data=data_set,
                                 label_to_classifier=LawBreakingLabels.KEY_CODE)
    decision_tree_classification(data_set=data_set,
                             label_to_classifier=LawBreakingLabels.KEY_CODE,
                             test_percentage=.2
                             )


def process_law_breaking_level(data_set: pd.DataFrame) -> None:
    data_set = preprocess_data_to_classifer(data=data_set,
                                 label_to_classifier=LawBreakingLabels.LAW_BREAKING_LEVEL)
    decision_tree_classification(data_set=data_set,
                             label_to_classifier=LawBreakingLabels.LAW_BREAKING_LEVEL,
                             test_percentage=.2
                             )
    
    plt.hist(data_set[LawBreakingLabels.LAW_BREAKING_LEVEL], bins=(data_set[LawBreakingLabels.LAW_BREAKING_LEVEL].nunique() * 2) - 1)
    plt.show()


def decision_tree_classification(data_set: pd.DataFrame,
                                 label_to_classifier: str,
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


if __name__ == "__main__":
    main()