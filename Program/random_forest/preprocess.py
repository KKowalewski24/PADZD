import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import pandas as pd
from module.label_names_mapper import *
from typing import Dict, List, Tuple


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = drop_unused_columns(data)
    data = remove_na(data)
    data = extract_hour_and_day(data)

    for column in ['day_of_week_sin','day_of_week_cos','day_of_year_sin','day_of_year_cos']:
        data = data[data[column].notna()]

    data = transform_labels(data)
    print("Data rows count, after preprocessing: ", data.shape[0])
    return data


def drop_unused_columns(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(columns=[IdentifierLabels.ID,
                              LawBreakingLabels.PD_CODE,
                              EventLocationLabels.PRECINCT_CODE,
                              EventLocationLabels.BOROUGH_NAME,
                              EventLocationLabels.LATITUDE,
                              EventLocationLabels.LONGITUDE,
                              # SuspectLabels.SUSPECT_AGE_GROUP,
                              # VictimLabels.VICTIM_AGE_GROUP,
                              DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_TIMESTAMP,
                              DateTimeEventLabels.EVENT_END_TIMESTAMP,
                              "CMPLNT_FR_TM", "CMPLNT_TO_TM", "OFNS_DESC",
                              'PD_DESC', 'JURIS_DESC', 'JURISDICTION_CODE',
                              'PARKS_NM', 'HADEVELOPT', 'HOUSING_PSA', 'X_COORD_CD', 'Y_COORD_CD', 'TRANSIT_DISTRICT',
                              'Lat_Lon', 'PATROL_BORO', 'STATION_NAME'
                              ])


def remove_na(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        DateTimeEventLabels.EVENT_START_TIMESTAMP,
        # DateTimeEventLabels.EVENT_END_TIMESTAMP,
        # DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_TIMESTAMP,
        LawBreakingLabels.KEY_CODE,
        LawBreakingLabels.LAW_BREAKING_LEVEL,
        EventStatusLabels.EVENT_STATUS,
        EventSurroundingsLabels.PLACE_TYPE,
        EventSurroundingsLabels.PLACE_TYPE_POSITION,
        SuspectLabels.SUSPECT_AGE_GROUP,
        SuspectLabels.SUSPECT_RACE,
        SuspectLabels.SUSPECT_SEX,
        VictimLabels.VICTIM_AGE_GROUP,
        VictimLabels.VICTIM_RACE,
        VictimLabels.VICTIM_SEX
    ]
    for column in columns:
        data = data[data[column].notna()]
    return data


def transform_labels(data: pd.DataFrame) -> pd.DataFrame:
    one_hot_columns = [
        EventStatusLabels.EVENT_STATUS,
        VictimLabels.VICTIM_RACE,
        VictimLabels.VICTIM_SEX,
        SuspectLabels.SUSPECT_RACE,
        SuspectLabels.SUSPECT_SEX
    ]
    ordinal_columns: List[Tuple[str, List]] = [
        (LawBreakingLabels.LAW_BREAKING_LEVEL, ["VIOLATION", "MISDEMEANOR", "FELONY"]),
        (SuspectLabels.SUSPECT_AGE_GROUP, ["<18", "18-24", "25-44", "45-64", "65+", "UNKNOWN"]),
        (VictimLabels.VICTIM_AGE_GROUP, ["<18", "18-24", "25-44", "45-64", "65+", "UNKNOWN"]),
    ]
    rest_columns = [
        # LawBreakingLabels.KEY_CODE,
        EventSurroundingsLabels.PLACE_TYPE,
        EventSurroundingsLabels.PLACE_TYPE_POSITION
    ]

    data = pd.get_dummies(data, columns=one_hot_columns, prefix=one_hot_columns)

    for ordinal_column in ordinal_columns:
        label, categories = ordinal_column
        data[label] = OrdinalEncoder(categories=[categories]).fit_transform(data[[label]])

    encoder = LabelEncoder()
    for column in rest_columns:
        data[column] = encoder.fit_transform(data[column])
    return data


def transform_date_and_time(data: pd.DataFrame, days_and_hours_sin_cos: pd.DataFrame) -> pd.DataFrame:
    data['day_of_week_sin'] = np.sin(days_and_hours_sin_cos['day_of_week'] * (2 * np.pi / 7))
    data['day_of_week_cos'] = np.cos(days_and_hours_sin_cos['day_of_week'] * (2 * np.pi / 7))
    data['day_of_year_sin'] = np.sin(days_and_hours_sin_cos['day_of_year'] * (2 * np.pi / 365))
    data['day_of_year_cos'] = np.cos(days_and_hours_sin_cos['day_of_year'] * (2 * np.pi / 365))
    return data


def extract_hour_and_day(data: pd.DataFrame) -> pd.DataFrame:
    temp_df = pd.DataFrame()
    temp_df['day_of_week'] = pd.to_datetime(data[DateTimeEventLabels.EVENT_START_TIMESTAMP],  errors = 'coerce') \
        .map(lambda date: date.weekday())
    temp_df['day_of_year'] = pd.to_datetime(data[DateTimeEventLabels.EVENT_START_TIMESTAMP],  errors = 'coerce').dt.dayofyear

    for column in ['day_of_week','day_of_year']:
        temp_df = temp_df[temp_df[column].notna()]
    return transform_date_and_time(data.drop(columns=[DateTimeEventLabels.EVENT_START_TIMESTAMP]), temp_df)
