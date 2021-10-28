from enum import Enum
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

from module.LabelNamesMapper import LabelNamesMapper
from module.utils import prepare_filename

RESULTS_DIR = "saved_plots/"


class ChartType(Enum):
    BAR = "Bar"
    LINE = "Line"


def plot_charts(df: pd.DataFrame, save_data: bool) -> None:
    # begin_datetime = ((df[LabelNamesMapper.date_time_event.EVENT_START_DATE] +
    #                    df[LabelNamesMapper.date_time_event.EVENT_START_TIME])
    #                   .apply(pd.to_datetime, format='%m/%d/%Y%H:%M:%S', errors='coerce'))
    # end_datetime = ((df[LabelNamesMapper.date_time_event.EVENT_END_DATE] +
    #                  df[LabelNamesMapper.date_time_event.EVENT_END_TIME])
    #                 .apply(pd.to_datetime, format='%m/%d/%Y%H:%M:%S', errors='coerce'))
    # print((end_datetime - begin_datetime).mean())
    # print((end_datetime - begin_datetime).std())
    # draw_hist(df, LabelNamesMapper.event_surroundings.PLACE_TYPE_POSITION, save_data)
    original_data_len = len(df.index)

    draw_hist(
        ((df[LabelNamesMapper.date_time_event.EVENT_START_DATE] +
          df[LabelNamesMapper.date_time_event.EVENT_START_TIME])
         .apply(pd.to_datetime, format='%m/%d/%Y%H:%M:%S', errors='coerce').dt.hour),
        original_data_len, ("Hours", "", ""), save_data
    )
    draw_hist(
        pd.to_datetime(
            df[LabelNamesMapper.date_time_event.EVENT_START_DATE], format='%m/%d/%Y', errors="coerce"
        ).dt.month,
        original_data_len, ("Months", "", ""), save_data
    )

    draw_hist(
        filter_na(df, LabelNamesMapper.event_surroundings.PLACE_TYPE_POSITION),
        original_data_len, ("Place type", "", ""), save_data
    )


def draw_hist(data: pd.DataFrame, original_data_len: int,
              description: Tuple[str, str, str], save_data: bool) -> None:
    plt.hist(data)
    set_descriptions(
        description[0] + calculate_sizes(original_data_len, len(data.index)),
        description[1], description[2]
    )
    show_and_save(description[0], save_data)


# def draw_hist(df: pd.DataFrame, column_name: str, save_data: bool) -> None:
#     df_filtered = df[df[column_name].notna()]
#     plt.hist(df_filtered[column_name])
#     df_len = len(df.index)
#     df_filtered_len = len(df_filtered.index)
#
#     plt.title(
#         f"{column_name}, liczba brakujących wartości: {df_len - df_filtered_len} "
#         f"({((df_len - df_filtered_len) / df_len) * 100}%) "
#     )
#     show_and_save(column_name, save_data)


def draw_plot(df: pd.DataFrame, x_axis_col_name: str,
              y_axis_col_name: str, chart_type: ChartType, save_data: bool) -> None:
    if chart_type == ChartType.LINE:
        plt.plot(df[x_axis_col_name], df[y_axis_col_name])
    elif chart_type == ChartType.BAR:
        plt.bar(df[x_axis_col_name], df[y_axis_col_name])

    plt.title(f"{x_axis_col_name} to {y_axis_col_name}")
    plt.xlabel(x_axis_col_name)
    plt.ylabel(y_axis_col_name)

    if save_data:
        plt.savefig(RESULTS_DIR + prepare_filename(f"{x_axis_col_name}#{y_axis_col_name}"))
        plt.close()
    plt.show()


def filter_na(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    return data[data[column_name].notna()][column_name]


def calculate_sizes(original_data_len: int, data_len: int) -> str:
    return (
        f" Number of missing values: {original_data_len - data_len} "
        f"({((original_data_len - data_len) / original_data_len) * 100}%) "
    )


def set_descriptions(title: str, x_label: str = "", y_label: str = "") -> None:
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def show_and_save(name: str, save_data: bool) -> None:
    if save_data:
        plt.savefig(RESULTS_DIR + prepare_filename(name))
        plt.close()
    plt.show()
