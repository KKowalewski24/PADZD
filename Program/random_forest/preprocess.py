import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import pandas as pd
from module.label_names_mapper import *
from typing import Dict, List, Tuple

PROCESSED_COLUMN_NAMES = [
    LawBreakingLabels.KEY_CODE,
    LawBreakingLabels.LAW_BREAKING_LEVEL,
    DateTimeEventLabels.EVENT_START_TIMESTAMP,
    # EventStatusLabels.EVENT_STATUS,
    # EventSurroundingsLabels.PLACE_TYPE,
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
    print("Data rows count, before preprocessing: ", data.shape[0])
    preprocess_age_and_sex(data)
    preprocess_place_type_position(data)

    print("Droping columns...")
    data = drop_unused_columns(data)

    print("Removing na...")
    data = remove_na(data)

    print("Extracting hour and day...")
    data = extract_hour_and_day(data)

    for column in ['day_of_week_sin', 'day_of_week_cos', 'day_of_year_sin', 'day_of_year_cos']:
        data = data[data[column].notna()]

    data = transform_labels(data)
    print("Data rows count, after preprocessing: ", data.shape[0])

    print(data.columns.values)
    return data


def preprocess_age_and_sex(df: pd.DataFrame) -> None:
    print("Imputating AGE GROUP by inserting most frequent value")
    age_groups: List[str] = ["(<18)", "(18-24)", "(25-44)", "(45-64)", "(65+)", "(UNKNOWN)"]

    df.loc[
        ~df[SuspectLabels.SUSPECT_AGE_GROUP].str.match(pat="|".join(age_groups), na=False),
        SuspectLabels.SUSPECT_AGE_GROUP
    ] = df[SuspectLabels.SUSPECT_AGE_GROUP].value_counts().idxmax()

    df.loc[
        ~df[VictimLabels.VICTIM_AGE_GROUP].str.match(pat="|".join(age_groups), na=False),
        VictimLabels.VICTIM_AGE_GROUP
    ] = df[VictimLabels.VICTIM_AGE_GROUP].value_counts().idxmax()

    print("Grouping RACE")
    df.loc[
        (df[SuspectLabels.SUSPECT_RACE] == "UNKNOWN") | (df[SuspectLabels.SUSPECT_RACE].isnull())
        | (~df[SuspectLabels.SUSPECT_RACE].str.contains("WHITE", na=False) & ~df[SuspectLabels.SUSPECT_RACE].str.contains("BLACK", na=False) & ~df[SuspectLabels.SUSPECT_RACE].str.contains("HISPANIC", na=False)),
        SuspectLabels.SUSPECT_RACE
    ] = "OTHER"
    df.loc[
        (df[SuspectLabels.SUSPECT_RACE] == "WHITE"),
        SuspectLabels.SUSPECT_RACE
    ] = "WHITE"
    df.loc[
        (df[SuspectLabels.SUSPECT_RACE] == "BLACK"),
        SuspectLabels.SUSPECT_RACE
    ] = "BLACK"
    df.loc[
        df[SuspectLabels.SUSPECT_RACE].str.contains("HISPANIC"),
        SuspectLabels.SUSPECT_RACE
    ] = "HISPANIC"

    df.loc[
        (df[VictimLabels.VICTIM_RACE] == "UNKNOWN") | (df[VictimLabels.VICTIM_RACE].isnull())
        | (~df[VictimLabels.VICTIM_RACE].str.contains("WHITE", na=False) & ~df[VictimLabels.VICTIM_RACE].str.contains("BLACK", na=False) & ~df[VictimLabels.VICTIM_RACE].str.contains("HISPANIC", na=False)),
        VictimLabels.VICTIM_RACE
    ] = "OTHER"
    df.loc[
        (df[VictimLabels.VICTIM_RACE] == "WHITE"),
        VictimLabels.VICTIM_RACE
    ] = "WHITE"
    df.loc[
        (df[VictimLabels.VICTIM_RACE] == "BLACK"),
        VictimLabels.VICTIM_RACE
    ] = "BLACK"
    df.loc[
        df[VictimLabels.VICTIM_RACE].str.contains("HISPANIC"),
        VictimLabels.VICTIM_RACE
    ] = "HISPANIC"


    print("Grouping GENDER")
    df.loc[df[SuspectLabels.SUSPECT_SEX] == "M", SuspectLabels.SUSPECT_SEX] = "MALE"
    df.loc[df[SuspectLabels.SUSPECT_SEX] == "F", SuspectLabels.SUSPECT_SEX] = "FEMALE"

    df.loc[
        ~df[SuspectLabels.SUSPECT_SEX].str.contains("F|M", na=False),
        SuspectLabels.SUSPECT_SEX
    ] = "OTHER"

    df.loc[df[VictimLabels.VICTIM_SEX] == "M", VictimLabels.VICTIM_SEX] = "MALE"
    df.loc[df[VictimLabels.VICTIM_SEX] == "F", VictimLabels.VICTIM_SEX] = "FEMALE"
    df.loc[
        ~df[VictimLabels.VICTIM_SEX].str.contains("F|M", na=False),
        VictimLabels.VICTIM_SEX
    ] = "OTHER"


def preprocess_place_type_position(df: pd.DataFrame) -> None:
    print("Grouping PLACE TYPE POSITION")
    df.loc[
        (df[EventSurroundingsLabels.PLACE_TYPE_POSITION].isnull()),
        EventSurroundingsLabels.PLACE_TYPE_POSITION
    ] = "UNKNOWN"
    df.loc[
        (~df[EventSurroundingsLabels.PLACE_TYPE_POSITION].str.contains("FRONT OF") & ~df[EventSurroundingsLabels.PLACE_TYPE_POSITION].str.contains("INSIDE") & ~df[EventSurroundingsLabels.PLACE_TYPE_POSITION].str.contains("UNKNOWN")),
        EventSurroundingsLabels.PLACE_TYPE_POSITION
    ] = "OTHER"


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
        # EventStatusLabels.EVENT_STATUS,
        VictimLabels.VICTIM_RACE,
        VictimLabels.VICTIM_SEX,
        SuspectLabels.SUSPECT_RACE,
        SuspectLabels.SUSPECT_SEX,
        # EventSurroundingsLabels.PLACE_TYPE,
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
