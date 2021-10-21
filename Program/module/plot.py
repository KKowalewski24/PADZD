from enum import Enum
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from module.LabelNamesMapper import LabelNamesMapper
from module.utils import create_directory, prepare_filename

RESULTS_DIR = "saved_plots/"


class ChartType(Enum):
    BAR = "Bar"
    LINE = "Line"


LINE_CHART_PARAM_SETUP: List[Tuple[str, str]] = [

]

BAR_CHART_PARAM_SETUP: List[Tuple[str, str]] = [
    (
        LabelNamesMapper.law_breaking.LAW_BREAKING_STATUS,
        LabelNamesMapper.law_breaking.KEY_CODE
    ),
    (
        LabelNamesMapper.victim_suspect.SUSPECT_RACE,
        LabelNamesMapper.victim_suspect.VICTIM_RACE,
    ),
    (
        LabelNamesMapper.victim_suspect.SUSPECT_SEX,
        LabelNamesMapper.victim_suspect.VICTIM_SEX,
    ),
    (
        LabelNamesMapper.victim_suspect.SUSPECT_AGE_GROUP,
        LabelNamesMapper.victim_suspect.VICTIM_AGE_GROUP,
    ),
    (
        LabelNamesMapper.location.PATROL_DISTRICT_NAME,
        LabelNamesMapper.law_breaking.KEY_CODE
    )
]


def plot_charts(dataset: pd.DataFrame, save_data: bool) -> None:
    # TODO MOVE THIS TO MORE APPROPRIATE PLACE
    label_encoder = LabelEncoder()
    for name in LabelNamesMapper.get_non_numeric_column_names():
        dataset[name] = label_encoder.fit_transform(dataset[name])

    create_directory(RESULTS_DIR)

    for params in LINE_CHART_PARAM_SETUP:
        draw_plot(dataset, params[0], params[1], ChartType.LINE, save_data)

    for params in BAR_CHART_PARAM_SETUP:
        draw_plot(dataset, params[0], params[1], ChartType.BAR, save_data)


def draw_plot(dataset: pd.DataFrame, x_axis_col_name: str,
              y_axis_col_name: str, chart_type: ChartType, save_data: bool) -> None:
    if chart_type == ChartType.LINE:
        plt.plot(dataset[x_axis_col_name], dataset[y_axis_col_name])
    elif chart_type == ChartType.BAR:
        plt.bar(dataset[x_axis_col_name], dataset[y_axis_col_name])

    plt.title(f"{x_axis_col_name} to {y_axis_col_name}")
    plt.xlabel(x_axis_col_name)
    plt.ylabel(y_axis_col_name)

    if save_data:
        plt.savefig(RESULTS_DIR + prepare_filename(f"{x_axis_col_name}#{y_axis_col_name}"))
        plt.close()
    plt.show()
