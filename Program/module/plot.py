from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from module.LabelNamesMapper import LabelNamesMapper
from module.utils import create_directory, prepare_filename

RESULTS_DIR = "saved_plots/"

PARAMS_SETUP: List[Tuple[str, str]] = [
    (
        LabelNamesMapper.law_breaking_labels.KEY_CODE,
        LabelNamesMapper.law_breaking_labels.LAW_BREAKING_STATUS
    ),
]


def plot_charts(dataset: pd.DataFrame, save_data: bool) -> None:
    # TODO MOVE THIS TO MORE APPROPRIATE PLACE
    label_encoder = LabelEncoder()
    for name in LabelNamesMapper.get_non_numeric_column_names():
        dataset[name] = label_encoder.fit_transform(dataset[name])

    create_directory(RESULTS_DIR)

    for params in PARAMS_SETUP:
        draw_plot(dataset, params[0], params[1], save_data)


def draw_plot(dataset: pd.DataFrame, x_axis_col_name: str,
              y_axis_col_name: str, save_data: bool) -> None:
    plt.plot(dataset[x_axis_col_name], dataset[y_axis_col_name])
    if save_data:
        plt.savefig(RESULTS_DIR + prepare_filename(f"{x_axis_col_name}#{y_axis_col_name}"))
        plt.close()
    plt.show()
