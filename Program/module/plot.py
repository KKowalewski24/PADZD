from enum import Enum
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from module.LabelNamesMapper import LabelNamesMapper
from module.utils import create_directory, prepare_filename

RESULTS_DIR = "saved_plots/"


class ChartType(Enum):
    BAR = "Bar"
    LINE = "Line"


class CalculationType(Enum):
    MEAN = "Mean"
    STD = "Std"


def generate_charts_stats(df: pd.DataFrame, save_data: bool) -> None:
    create_directory(RESULTS_DIR)
    original_data_len = len(df.index)

    draw_hist(
        filter_na(df, LabelNamesMapper.date_time_event.EVENT_START_TIME)
            .astype(str).str[:2].sort_values(),
        original_data_len, ("Hours", "", ""), save_data
    )
    draw_hist(
        pd.to_datetime(
            df[LabelNamesMapper.date_time_event.EVENT_START_DATE], format='%m/%d/%Y', errors="coerce"
        ).dt.month,
        original_data_len, ("Months", "", ""), save_data
    )
    draw_hist(
        filter_na(df, LabelNamesMapper.law_breaking.LAW_BREAKING_LEVEL).astype(str),
        original_data_len, ("Law breaking level", "", ""), save_data
    )
    draw_hist(
        filter_na(df, LabelNamesMapper.event_status.EVENT_STATUS).astype(str),
        original_data_len, ("Event status", "", ""), save_data
    )
    draw_hist(
        filter_na(df, LabelNamesMapper.event_surroundings.PLACE_TYPE_POSITION).astype(str),
        original_data_len, ("Place type", "", ""), save_data
    )

    # Stats between start time of event and end time
    full_time_data = df[
        df[LabelNamesMapper.date_time_event.EVENT_END_DATE].notna()
        & df[LabelNamesMapper.date_time_event.EVENT_START_TIME].notna()
        ]

    begin_datetime = ((full_time_data[LabelNamesMapper.date_time_event.EVENT_START_DATE] +
                       full_time_data[LabelNamesMapper.date_time_event.EVENT_START_TIME])
                      .apply(pd.to_datetime, format='%m/%d/%Y%H:%M:%S', errors='coerce'))
    end_datetime = ((full_time_data[LabelNamesMapper.date_time_event.EVENT_END_DATE] +
                     full_time_data[LabelNamesMapper.date_time_event.EVENT_END_TIME])
                    .apply(pd.to_datetime, format='%m/%d/%Y%H:%M:%S', errors='coerce'))

    print(
        f"Number of rows with {LabelNamesMapper.date_time_event.EVENT_END_DATE} and "
        f"{LabelNamesMapper.date_time_event.EVENT_END_TIME}",
        len(full_time_data.index)
    )
    print(calculate(
        begin_datetime, end_datetime,
        [
            LabelNamesMapper.date_time_event.EVENT_START_DATE,
            LabelNamesMapper.date_time_event.EVENT_START_TIME
        ],
        [
            LabelNamesMapper.date_time_event.EVENT_END_DATE,
            LabelNamesMapper.date_time_event.EVENT_END_TIME
        ],
        CalculationType.MEAN))
    print(calculate(
        begin_datetime, end_datetime,
        [
            LabelNamesMapper.date_time_event.EVENT_START_DATE,
            LabelNamesMapper.date_time_event.EVENT_START_TIME
        ],
        [
            LabelNamesMapper.date_time_event.EVENT_END_DATE,
            LabelNamesMapper.date_time_event.EVENT_END_TIME
        ],
        CalculationType.STD))

    # Stats between start time of event and submission to police
    begin_date = pd.to_datetime(
        df[LabelNamesMapper.date_time_event.EVENT_START_DATE], format='%m/%d/%Y', errors="coerce"
    )
    submission_date = pd.to_datetime(
        df[LabelNamesMapper.date_time_submission.SUBMISSION_TO_POLICE_DATE],
        format='%m/%d/%Y', errors="coerce"
    )
    print(calculate(begin_date, submission_date,
                    [LabelNamesMapper.date_time_event.EVENT_START_DATE],
                    [LabelNamesMapper.date_time_submission.SUBMISSION_TO_POLICE_DATE],
                    CalculationType.MEAN))
    print(calculate(begin_date, submission_date,
                    [LabelNamesMapper.date_time_event.EVENT_START_DATE],
                    [LabelNamesMapper.date_time_submission.SUBMISSION_TO_POLICE_DATE],
                    CalculationType.STD))


def draw_hist(data: pd.DataFrame, original_data_len: int,
              description: Tuple[str, str, str], save_data: bool) -> None:
    plt.hist(data, bins=(data.nunique() * 2) - 1)
    set_descriptions(
        description[0] + ", " + calculate_sizes(original_data_len, len(data.index)),
        description[1], description[2]
    )
    show_and_save(description[0], save_data)


def draw_plot(data_x_axis: pd.Series, data_y_axis: pd.Series, x_axis_col_name: str,
              y_axis_col_name: str, chart_type: ChartType, save_data: bool) -> None:
    if chart_type == ChartType.LINE:
        plt.plot(data_x_axis, data_y_axis)
    elif chart_type == ChartType.BAR:
        plt.bar(data_x_axis, data_y_axis)

    set_descriptions(f"{x_axis_col_name} to {y_axis_col_name}", x_axis_col_name, y_axis_col_name)
    show_and_save(f"{x_axis_col_name}#{y_axis_col_name}", save_data)


def calculate(begin_value, end_value, begin_labels: List[str],
              end_labels: List[str], calculation_type: CalculationType) -> str:
    title = ""
    value = 0
    if CalculationType.MEAN == calculation_type:
        title = "Mean of"
        value = (end_value - begin_value).mean()
    elif CalculationType.STD == calculation_type:
        title = "Std of"
        value = (end_value - begin_value).std()

    return (f"{title} {[label + ' ' for label in begin_labels]} "
            f"and {[label + ' ' for label in end_labels]} {value}")


def filter_na(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    return data[data[column_name].notna()][column_name]


def calculate_sizes(original_data_len: int, data_len: int) -> str:
    return (
        f" Number of missing values: {original_data_len - data_len} "
        f"({round(((original_data_len - data_len) / original_data_len) * 100, 2)}%) "
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
