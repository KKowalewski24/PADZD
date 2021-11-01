from enum import Enum
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

from module.utils import prepare_filename

RESULTS_DIR = "saved_plots/"


class ChartType(Enum):
    BAR = "Bar"
    LINE = "Line"


class CalculationType(Enum):
    MEAN = "Mean"
    STD = "Std"


def draw_plot(data_x_axis: pd.Series, data_y_axis: pd.Series, x_axis_col_name: str,
              y_axis_col_name: str, chart_type: ChartType, save_data: bool) -> None:
    if chart_type == ChartType.LINE:
        plt.plot(data_x_axis, data_y_axis)
    elif chart_type == ChartType.BAR:
        plt.bar(data_x_axis, data_y_axis)

    _set_descriptions(f"{x_axis_col_name} to {y_axis_col_name}", x_axis_col_name, y_axis_col_name)
    _show_and_save(f"{x_axis_col_name}-{y_axis_col_name}", save_data)


def draw_hist(data: pd.DataFrame, original_data_len: int,
              description: Tuple[str, str, str], save_data: bool) -> None:
    plt.hist(data, bins=(data.nunique() * 2) - 1)
    _set_descriptions(
        description[0] + ", " + _calculate_sizes(original_data_len, len(data.index)),
        description[1], description[2]
    )
    _show_and_save(description[0], save_data)


def _calculate_sizes(original_data_len: int, data_len: int) -> str:
    return (
        f" Number of missing values: {original_data_len - data_len} "
        f"({round(((original_data_len - data_len) / original_data_len) * 100, 2)}%) "
    )


def _filter_na(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    return data[data[column_name].notna()][column_name]


def _set_descriptions(title: str, x_label: str = "", y_label: str = "") -> None:
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def _show_and_save(name: str, save_data: bool) -> None:
    if save_data:
        plt.savefig(RESULTS_DIR + prepare_filename(name))
        plt.close()
    plt.show()
