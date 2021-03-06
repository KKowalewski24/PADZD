import json
import logging
import os
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder

"""
How to run:
    python dataset_preprocessing.py -f ../data/NYPD_Complaint_Data_Historic.csv
"""

JSON = ".json"
CSV = ".csv"
RESULTS_DIR = "preprocessing_output/"
LOGS_FILENAME: str = "app.log"

logging.basicConfig(
    filename=LOGS_FILENAME,
    level=logging.INFO,
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} - %(message)s"
)


def main() -> None:
    args = prepare_args()
    filepath = args.filepath
    create_directory(RESULTS_DIR)

    display_and_log("Loading data")
    df: pd.DataFrame = pd.read_csv(filepath)
    display_and_log(f"Size of loaded data: {len(df.index)}")

    merge_cols(df)
    df.fillna(df.mean(), inplace=True)
    group_count_rename_data(df)
    drop_cols(df)
    columns_number = len(df.columns)
    calculate_stats(df)
    print_unique_ordinal_values(df)
    df = encode_columns(df)
    df_reduced = reduce_dimensions(df, columns_number)
    df_reduced_2_times_more = reduce_dimensions(df, columns_number * 2)

    display_and_log("Saving data to file")
    df.to_csv(
        RESULTS_DIR + prepare_filename("NYPD_Data_Preprocessed", CSV), index=False
    )
    df_reduced.to_csv(
        RESULTS_DIR + prepare_filename("NYPD_Data_Preprocessed_Reduced", CSV), index=False
    )
    df_reduced_2_times_more.to_csv(
        RESULTS_DIR + prepare_filename("NYPD_Data_Preprocessed_Reduced_2", CSV), index=False
    )

    display_finish()


def merge_cols(df: pd.DataFrame) -> None:
    display_and_log(f"Merging {DateTimeEventLabels.EVENT_START_TIMESTAMP}")
    df[DateTimeEventLabels.EVENT_START_TIMESTAMP] = (
        (df[DateTimeEventLabels.EVENT_START_DATE] + df[DateTimeEventLabels.EVENT_START_TIME])
            .apply(pd.to_datetime, format='%m/%d/%Y%H:%M:%S', errors='coerce')
    )
    df = _convert_datetime_to_timestamp(
        df, DateTimeEventLabels.EVENT_START_TIMESTAMP, DateTimeEventLabels.EVENT_START_TIMESTAMP_UNIX
    )

    display_and_log(f"Merging {DateTimeEventLabels.EVENT_END_TIMESTAMP}")
    df[DateTimeEventLabels.EVENT_END_TIMESTAMP] = (
        (df[DateTimeEventLabels.EVENT_END_DATE] + df[DateTimeEventLabels.EVENT_END_TIME])
            .apply(pd.to_datetime, format='%m/%d/%Y%H:%M:%S', errors='coerce')
    )
    df = _convert_datetime_to_timestamp(
        df, DateTimeEventLabels.EVENT_END_TIMESTAMP, DateTimeEventLabels.EVENT_END_TIMESTAMP_UNIX
    )

    display_and_log(f"Merging {DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_TIMESTAMP}")
    df[DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_TIMESTAMP] = pd.to_datetime(
        df[DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_DATE]
    )
    df = _convert_datetime_to_timestamp(
        df, DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_TIMESTAMP,
        DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_TIMESTAMP_UNIX
    )


def _convert_datetime_to_timestamp(df: pd.DataFrame, input_column: str,
                                   output_column: str) -> pd.DataFrame:
    df[output_column] = pd.to_datetime(df[input_column]).values.astype(np.int64) // 10 ** 9
    return df


def group_count_rename_data(df: pd.DataFrame) -> None:
    display_and_log(f"Grouping {LawBreakingLabels.KEY_CODE}")
    (df.groupby(LawBreakingLabels.KEY_CODE)[LawBreakingLabels.OFFENSE_DESCRIPTION]
     .unique()
     .reset_index()
     .to_csv(RESULTS_DIR + prepare_filename("key_code_desc_map", CSV), index=False))

    display_and_log(f"Grouping {LawBreakingLabels.PD_CODE}")
    (df.groupby(LawBreakingLabels.PD_CODE)[LawBreakingLabels.PD_DESCRIPTION]
     .unique()
     .reset_index()
     .to_csv(RESULTS_DIR + prepare_filename("pd_code_desc_map", CSV), index=False))

    display_and_log(f"Counting {SuspectLabels.SUSPECT_AGE_GROUP}")
    (df.groupby([SuspectLabels.SUSPECT_AGE_GROUP])[IdentifierLabels.ID]
     .count()
     .reset_index()
     .to_csv(RESULTS_DIR + prepare_filename("suspect_age_group_count", CSV), index=False))

    display_and_log(f"Counting {VictimLabels.VICTIM_AGE_GROUP}")
    (df.groupby([VictimLabels.VICTIM_AGE_GROUP])[IdentifierLabels.ID]
     .count()
     .reset_index()
     .to_csv(RESULTS_DIR + prepare_filename("victim_age_group_count", CSV), index=False))

    display_and_log("Imputating AGE GROUP by inserting most frequent value")
    age_groups: List[str] = ["(<18)", "(18-24)", "(25-44)", "(45-64)", "(65+)", "(UNKNOWN)"]

    df.loc[
        ~df[SuspectLabels.SUSPECT_AGE_GROUP].str.match(pat="|".join(age_groups), na=False),
        SuspectLabels.SUSPECT_AGE_GROUP
    ] = df[SuspectLabels.SUSPECT_AGE_GROUP].value_counts().idxmax()

    df.loc[
        ~df[VictimLabels.VICTIM_AGE_GROUP].str.match(pat="|".join(age_groups), na=False),
        VictimLabels.VICTIM_AGE_GROUP
    ] = df[VictimLabels.VICTIM_AGE_GROUP].value_counts().idxmax()

    display_and_log("Grouping RACE")
    df.loc[
        (df[SuspectLabels.SUSPECT_RACE] == "UNKNOWN") | (df[SuspectLabels.SUSPECT_RACE].isnull()),
        SuspectLabels.SUSPECT_RACE
    ] = "OTHER"

    df.loc[
        (df[VictimLabels.VICTIM_RACE] == "UNKNOWN") | (df[VictimLabels.VICTIM_RACE].isnull()),
        VictimLabels.VICTIM_RACE
    ] = "OTHER"

    display_and_log("Grouping GENDER")
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


def drop_cols(df: pd.DataFrame) -> None:
    display_and_log("Dropping cols")
    df.drop(columns=[
        DateTimeEventLabels.EVENT_START_DATE,
        DateTimeEventLabels.EVENT_START_TIME,
        DateTimeEventLabels.EVENT_START_TIMESTAMP,
        DateTimeEventLabels.EVENT_END_DATE,
        DateTimeEventLabels.EVENT_END_TIME,
        DateTimeEventLabels.EVENT_END_TIMESTAMP,
        DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_DATE,
        DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_TIMESTAMP,

        LawBreakingLabels.OFFENSE_DESCRIPTION,
        LawBreakingLabels.PD_DESCRIPTION,

        EventLocationLabels.JURISDICTION_DESCRIPTION,
        EventLocationLabels.JURISDICTION_CODE,
        EventLocationLabels.PARK_NAME,
        EventLocationLabels.NYC_HOUSING_DEVELOPMENT,
        EventLocationLabels.DEVELOPMENT_LEVEL_CODE,
        EventLocationLabels.NYC_X_COORDINATE,
        EventLocationLabels.NYC_Y_COORDINATE,
        EventLocationLabels.TRANSIT_DISTRICT_CODE,
        EventLocationLabels.LATITUDE_LONGITUDE,
        EventLocationLabels.PATROL_DISTRICT_NAME,
        EventLocationLabels.TRANSIT_STATION_NAME,
    ], inplace=True)


def calculate_stats(df: pd.DataFrame) -> None:
    columns: List[str] = [
        DateTimeEventLabels.EVENT_START_TIMESTAMP_UNIX,
        DateTimeEventLabels.EVENT_END_TIMESTAMP_UNIX,
        DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_TIMESTAMP_UNIX,
        LawBreakingLabels.KEY_CODE,
        LawBreakingLabels.PD_CODE,
        LawBreakingLabels.LAW_BREAKING_LEVEL,
        EventStatusLabels.EVENT_STATUS,
        EventSurroundingsLabels.PLACE_TYPE,
        EventSurroundingsLabels.PLACE_TYPE_POSITION,
        EventLocationLabels.PRECINCT_CODE,
        EventLocationLabels.BOROUGH_NAME,
        EventLocationLabels.LATITUDE,
        EventLocationLabels.LONGITUDE,
        SuspectLabels.SUSPECT_AGE_GROUP,
        SuspectLabels.SUSPECT_RACE,
        SuspectLabels.SUSPECT_SEX,
        VictimLabels.VICTIM_AGE_GROUP,
        VictimLabels.VICTIM_RACE,
        VictimLabels.VICTIM_SEX
    ]

    display_and_log("Calculating stats")
    stats: Dict[str, int] = {}
    for column in columns:
        stats[column] = df[column].isna().sum().astype(str)

    with open(RESULTS_DIR + prepare_filename("missing_values", JSON), "w") as file:
        json.dump(stats, file)


def print_unique_ordinal_values(df: pd.DataFrame) -> None:
    print(df[LawBreakingLabels.LAW_BREAKING_LEVEL].unique())
    print(df[SuspectLabels.SUSPECT_AGE_GROUP].unique())
    print(df[VictimLabels.VICTIM_AGE_GROUP].unique())


def encode_columns(df: pd.DataFrame) -> pd.DataFrame:
    one_hot_columns = [
        LawBreakingLabels.KEY_CODE,
        LawBreakingLabels.PD_CODE,
        EventStatusLabels.EVENT_STATUS,
        EventSurroundingsLabels.PLACE_TYPE,
        EventSurroundingsLabels.PLACE_TYPE_POSITION,
        EventLocationLabels.PRECINCT_CODE,
        EventLocationLabels.BOROUGH_NAME,
        SuspectLabels.SUSPECT_RACE,
        SuspectLabels.SUSPECT_SEX,
        VictimLabels.VICTIM_RACE,
        VictimLabels.VICTIM_SEX,
    ]
    display_and_log("Encoding one hot columns")
    df = pd.get_dummies(df, columns=one_hot_columns, prefix=one_hot_columns)

    ordinal_columns: List[Tuple[str, List]] = [
        (LawBreakingLabels.LAW_BREAKING_LEVEL, ["VIOLATION", "MISDEMEANOR", "FELONY"]),
        (SuspectLabels.SUSPECT_AGE_GROUP, ["<18", "18-24", "25-44", "45-64", "65+", "UNKNOWN"]),
        (VictimLabels.VICTIM_AGE_GROUP, ["<18", "18-24", "25-44", "45-64", "65+", "UNKNOWN"]),
    ]
    display_and_log("Encoding ordinal columns")
    for ordinal_column in ordinal_columns:
        label, categories = ordinal_column
        df[label] = OrdinalEncoder(categories=categories).fit_transform(df[[label]])

    return df


def reduce_dimensions(df: pd.DataFrame, output_dim_number: int) -> pd.DataFrame:
    display_and_log(f"Reduce dimensions for number of target dims: {output_dim_number}")
    return pd.DataFrame(PCA(n_components=output_dim_number).fit_transform(df))


def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-f", "--filepath", required=True, type=str, help="Filepath to CSV file"
    )

    return arg_parser.parse_args()


def display_and_log(text: str) -> None:
    print(text)
    logging.info(text)


def prepare_filename(name: str, extension: str = "", add_date: bool = True) -> str:
    return (name + ("-" + datetime.now().strftime("%H%M%S") if add_date else "")
            + extension).replace(" ", "")


def create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def check_types() -> None:
    subprocess.call(["mypy", "."])


def check_if_exists_in_args(arg: str) -> bool:
    return arg in sys.argv


def display_finish() -> None:
    display_and_log("------------------------------------------------------------------------")
    display_and_log("FINISHED")
    display_and_log("------------------------------------------------------------------------")


class IdentifierLabels:
    ID = "CMPLNT_NUM"


class DateTimeEventLabels:
    EVENT_START_DATE = "CMPLNT_FR_DT"
    EVENT_START_TIME = "CMPLNT_FR_TM"
    EVENT_START_TIMESTAMP = "CMPLNT_FR_TIMESTAMP"
    EVENT_START_TIMESTAMP_UNIX = "CMPLNT_FR_TIMESTAMP_UNIX"
    EVENT_END_DATE = "CMPLNT_TO_DT"
    EVENT_END_TIME = "CMPLNT_TO_TM"
    EVENT_END_TIMESTAMP = "CMPLNT_TO_TIMESTAMP"
    EVENT_END_TIMESTAMP_UNIX = "CMPLNT_TO_TIMESTAMP_UNIX"


class DateTimeSubmissionLabels:
    SUBMISSION_TO_POLICE_DATE = "RPT_DT"
    SUBMISSION_TO_POLICE_TIMESTAMP = "RPT_TIMESTAMP"
    SUBMISSION_TO_POLICE_TIMESTAMP_UNIX = "RPT_TIMESTAMP_UNIX"


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
    DEVELOPMENT_LEVEL_CODE = "HOUSING_PSA"
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


if __name__ == "__main__":
    if check_if_exists_in_args("-t"):
        check_types()
    else:
        main()
