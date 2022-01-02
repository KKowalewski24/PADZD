import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import pandas as pd
from module.label_names_mapper import *
from typing import Dict, List, Tuple

PROCESSED_COLUMN_NAMES = [
    LawBreakingLabels.KEY_CODE,
    LawBreakingLabels.LAW_BREAKING_LEVEL,
    DateTimeEventLabels.EVENT_START_TIMESTAMP,
    EventStatusLabels.EVENT_STATUS,
    EventSurroundingsLabels.PLACE_TYPE,
    EventSurroundingsLabels.PLACE_TYPE_POSITION,
    SuspectLabels.SUSPECT_AGE_GROUP,
    SuspectLabels.SUSPECT_RACE,
    SuspectLabels.SUSPECT_SEX,
    VictimLabels.VICTIM_AGE_GROUP,
    VictimLabels.VICTIM_RACE,
    VictimLabels.VICTIM_SEX,
    # EventLocationLabels.LONGITUDE,
    # EventLocationLabels.LATITUDE
]

#Dla 100k testowe accuracy
#victim i suspect i czas
    # train_acc: 0.7095739430827439 	test_acc: 0.2937228582708251
#suspect i czas
    # train_acc: 0.34135548609968747 	test_acc: 0.21713383339913148
#victim i czas
    # train_acc: 0.37366552491436855 	test_acc: 0.25566278313915697
#victim i suspect
    # train_acc: 0.40294112809040517 	test_acc: 0.3469307191262583
#victim i suspect LAW_BREAKING_LEVEL
    # train_acc: 0.6643363545145 	test_acc: 0.6086584643726561
#victim i suspect EVENT_STATUS
    # train_acc: 0.41207046863948155 	test_acc: 0.34719389433515363
#victim i suspect EVENT_STATUS PLACE_TYPE  i PLACE_TYPE_POSITION
#   train_acc: 0.6027801842955028 	test_acc: 0.35667034178610807
#     train_acc: 0.6075057100102387 	test_acc: 0.35659158922664985 HOTENCODING
#victim i suspect EVENT_STATUS PLACE_TYPE
    # train_acc: 0.5468540940550719 	test_acc: 0.3465477370333664
    # train_acc: 0.5465402467830655 	test_acc: 0.34978526593987447 HOTencoding
#victim i suspect EVENT_STATUS PLACE_TYPE_POSITION
    # train_acc: 0.47011678946543345 	test_acc: 0.3623608716099702
    # train_acc: 0.47141009562627373 	test_acc: 0.3617338140774416 hotencoding

# victim i suspect PLACE_TYPE_POSITION
#     train_acc: 0.4628860322934629 	test_acc: 0.3599310236714219
#     train_acc: 0.4619258504467785 	test_acc: 0.36361498667502745 hotencofing

# dodanie longitude i latitude
# train_acc: 0.9768932267168391 	test_acc: 0.3387425525243023


def preprocess_data_to_classifer(data: pd.DataFrame, label_to_classifier: str) -> pd.DataFrame:
    if label_to_classifier == LawBreakingLabels.KEY_CODE:
        data = data[data[LawBreakingLabels.KEY_CODE].notna()]
        data.drop(LawBreakingLabels.LAW_BREAKING_LEVEL, axis=1, inplace=True)
    else:
        data = data[data[LawBreakingLabels.LAW_BREAKING_LEVEL].notna()]
        data.drop(LawBreakingLabels.KEY_CODE, axis=1, inplace=True)
    return data


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
    # removes all columns that are NOT in the given list
    return data[data.columns.intersection(PROCESSED_COLUMN_NAMES)]


def remove_na(data: pd.DataFrame) -> pd.DataFrame:
    for column in PROCESSED_COLUMN_NAMES:
        if column != LawBreakingLabels.KEY_CODE or column != LawBreakingLabels.LAW_BREAKING_LEVEL:
            data = data[data[column].notna()]
    return data


def transform_labels(data: pd.DataFrame) -> pd.DataFrame:
    one_hot_columns = [
        EventStatusLabels.EVENT_STATUS,
        VictimLabels.VICTIM_RACE,
        VictimLabels.VICTIM_SEX,
        SuspectLabels.SUSPECT_RACE,
        SuspectLabels.SUSPECT_SEX,
        EventSurroundingsLabels.PLACE_TYPE,
        EventSurroundingsLabels.PLACE_TYPE_POSITION
    ]
    ordinal_columns: List[Tuple[str, List]] = [
        # (LawBreakingLabels.LAW_BREAKING_LEVEL, ["VIOLATION", "MISDEMEANOR", "FELONY"]),
        (VictimLabels.VICTIM_AGE_GROUP, ["<18", "18-24", "25-44", "45-64", "65+", "UNKNOWN"]),
        (SuspectLabels.SUSPECT_AGE_GROUP, ["<18", "18-24", "25-44", "45-64", "65+", "UNKNOWN"]),
    ]
    rest_columns = [
        # LawBreakingLabels.KEY_CODE,
        # EventSurroundingsLabels.PLACE_TYPE,
        # EventSurroundingsLabels.PLACE_TYPE_POSITION
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