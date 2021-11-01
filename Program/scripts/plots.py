import os
from datetime import datetime
from enum import Enum
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


class IdentifierLabels:
    ID = "CMPLNT_NUM"


class DateTimeEventLabels:
    EVENT_START_DATE = "CMPLNT_FR_DT"
    EVENT_START_TIME = "CMPLNT_FR_TM"
    EVENT_START_TIMESTAMP = "CMPLNT_FR_TIMESTAMP"
    EVENT_END_DATE = "CMPLNT_TO_DT"
    EVENT_END_TIME = "CMPLNT_TO_TM"
    EVENT_DATE_TIMESTAMP = "CMPLNT_TO_TIMESTAMP"


class DateTimeSubmissionLabels:
    SUBMISSION_TO_POLICE_DATE = "RPT_DT"
    SUBMISSION_TO_POLICE_TIMESTAMP = "RPT_TIMESTAMP"


class LawBreakingLabels:
    KEY_CODE = "KY_CD"
    OFFENSE_DESCRIPTION = "OFNS_DESC"
    PD_CODE = "PD_CD"
    PD_DESCRIPTION = "PD_DESC"
    LAW_BREAKING_LEVEL = "LAW_CAT_CD"


class EventStatusLabels:
    EVENT_STATUS = "CRM_ATPT_CPTD_CD"


class EventSurroundingsLabels:
    PLACE_TYPE = "PREM_TYP_DESC"
    PLACE_TYPE_POSITION = "LOC_OF_OCCUR_DESC"


class EventLocationLabels:
    PRECINCT_CODE = "ADDR_PCT_CD"
    BOROUGH_NAME = "BORO_NM"
    JURISDICTION_DESCRIPTION = "JURIS_DESC"
    JURISDICTION_CODE = "JURISDICTION_CODE"
    PARK_NAME = "PARKS_NM"
    NYC_HOUSING_DEVELOPMENT = "HADEVELOPT"
    DEVELOPMENT_LEVEl_CODE = "HOUSING_PSA"
    NYC_X_COORDINATE = "X_COORD_CD"
    NYC_Y_COORDINATE = "Y_COORD_CD"
    TRANSIT_DISTRICT_CODE = "TRANSIT_DISTRICT"
    LATITUDE = "Latitude"
    LONGITUDE = "Longitude"
    LATITUDE_LONGITUDE = "Lat_Lon"
    PATROL_DISTRICT_NAME = "PATROL_BORO"
    TRANSIT_STATION_NAME = "STATION_NAME"


class SuspectLabels:
    SUSPECT_AGE_GROUP = "SUSP_AGE_GROUP"
    SUSPECT_RACE = "SUSP_RACE"
    SUSPECT_SEX = "SUSP_SEX"


class VictimLabels:
    VICTIM_AGE_GROUP = "VIC_AGE_GROUP"
    VICTIM_RACE = "VIC_RACE"
    VICTIM_SEX = "VIC_SEX"


class LabelNamesMapper:
    identifier: IdentifierLabels = IdentifierLabels
    date_time_event: DateTimeEventLabels = DateTimeEventLabels
    date_time_submission: DateTimeSubmissionLabels = DateTimeSubmissionLabels
    law_breaking: LawBreakingLabels = LawBreakingLabels
    event_status: EventStatusLabels = EventStatusLabels
    event_surroundings: EventSurroundingsLabels = EventSurroundingsLabels
    event_location: EventLocationLabels = EventLocationLabels
    suspect: SuspectLabels = SuspectLabels
    victim: VictimLabels = VictimLabels


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
        _filter_na(df, LabelNamesMapper.date_time_event.EVENT_START_TIME)
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
        _filter_na(df, LabelNamesMapper.law_breaking.LAW_BREAKING_LEVEL).astype(str),
        original_data_len, ("Law breaking level", "", ""), save_data
    )
    draw_hist(
        _filter_na(df, LabelNamesMapper.event_status.EVENT_STATUS).astype(str),
        original_data_len, ("Event status", "", ""), save_data
    )
    draw_hist(
        _filter_na(df, LabelNamesMapper.event_surroundings.PLACE_TYPE_POSITION).astype(str),
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
    print(_calculate_measures(
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
    print(_calculate_measures(
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
    print(_calculate_measures(begin_date, submission_date,
                              [LabelNamesMapper.date_time_event.EVENT_START_DATE],
                              [LabelNamesMapper.date_time_submission.SUBMISSION_TO_POLICE_DATE],
                              CalculationType.MEAN))
    print(_calculate_measures(begin_date, submission_date,
                              [LabelNamesMapper.date_time_event.EVENT_START_DATE],
                              [LabelNamesMapper.date_time_submission.SUBMISSION_TO_POLICE_DATE],
                              CalculationType.STD))


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


def _calculate_measures(begin_value, end_value, begin_labels: List[str],
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


def create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def prepare_filename(name: str, extension: str = "", add_date: bool = True) -> str:
    return (name + ("-" + datetime.now().strftime("%H%M%S") if add_date else "")
            + extension).replace(" ", "")


if __name__ == '__main__':
    generate_charts_stats(pd.read_csv("../data/NYPD_Complaint_Data_Historic.csv"), True)
