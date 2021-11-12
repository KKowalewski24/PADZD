import json
import os
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from typing import Dict, List

import pandas as pd

"""
How to run:
    python dataset_preprocessing.py -f ../data/NYPD_Complaint_Data_Historic.csv
"""

RESULTS_DIR = "preprocessing_output/"


def main() -> None:
    args = prepare_args()
    filepath = args.filepath
    create_directory(RESULTS_DIR)

    print("Loading data...")
    df: pd.DataFrame = pd.read_csv(filepath)
    print("Size of loaded data:", len(df.index))

    merge_cols(df)
    drop_cols(df)
    group_data(df)

    calculate_stats(
        df,
        [
            DateTimeEventLabels.EVENT_START_TIMESTAMP,
            DateTimeEventLabels.EVENT_END_TIMESTAMP,
            DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_TIMESTAMP,
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
    )

    print("Saving data to file...")
    df.to_csv(f"{RESULTS_DIR}NYPD_Data_Preprocessed.csv", index=False)

    display_finish()


def merge_cols(df: pd.DataFrame) -> None:
    print(f"Merging {DateTimeEventLabels.EVENT_START_TIMESTAMP}")
    df[DateTimeEventLabels.EVENT_START_TIMESTAMP] = (
        (df[DateTimeEventLabels.EVENT_START_DATE] + df[DateTimeEventLabels.EVENT_START_TIME])
            .apply(pd.to_datetime, format='%m/%d/%Y%H:%M:%S', errors='coerce')
    )

    print(f"Merging {DateTimeEventLabels.EVENT_END_TIMESTAMP}")
    df[DateTimeEventLabels.EVENT_END_TIMESTAMP] = (
        (df[DateTimeEventLabels.EVENT_END_DATE] + df[DateTimeEventLabels.EVENT_END_TIME])
            .apply(pd.to_datetime, format='%m/%d/%Y%H:%M:%S', errors='coerce')
    )

    print(f"Merging {DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_TIMESTAMP}")
    df[DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_TIMESTAMP] = pd.to_datetime(
        df[DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_DATE]
    )


def drop_cols(df: pd.DataFrame) -> None:
    print("Dropping cols")
    df.drop(columns=[
        DateTimeEventLabels.EVENT_START_DATE,
        DateTimeEventLabels.EVENT_START_TIME,
        DateTimeEventLabels.EVENT_END_DATE,
        DateTimeEventLabels.EVENT_END_TIME,
        DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_DATE,

        LawBreakingLabels.OFFENSE_DESCRIPTION,
        LawBreakingLabels.PD_DESCRIPTION,

        EventLocationLabels.JURISDICTION_DESCRIPTION,
        EventLocationLabels.JURISDICTION_CODE,
        EventLocationLabels.PARK_NAME,
        EventLocationLabels.NYC_HOUSING_DEVELOPMENT,
        EventLocationLabels.DEVELOPMENT_LEVEl_CODE,
        EventLocationLabels.NYC_X_COORDINATE,
        EventLocationLabels.NYC_Y_COORDINATE,
        EventLocationLabels.TRANSIT_DISTRICT_CODE,
        EventLocationLabels.LATITUDE_LONGITUDE,
        EventLocationLabels.PATROL_DISTRICT_NAME,
        EventLocationLabels.TRANSIT_STATION_NAME,
    ], inplace=True)


def group_data(df: pd.DataFrame) -> None:
    print(f"Grouping {LawBreakingLabels.KEY_CODE}")
    (df.groupby(LawBreakingLabels.KEY_CODE)[LawBreakingLabels.OFFENSE_DESCRIPTION]
     .unique()
     .reset_index()
     .to_csv(f"{RESULTS_DIR}key_code_desc_map.csv", index=False))

    print(f"Grouping {LawBreakingLabels.PD_CODE}")
    (df.groupby(LawBreakingLabels.PD_CODE)[LawBreakingLabels.PD_DESCRIPTION]
     .unique()
     .reset_index()
     .to_csv(f"{RESULTS_DIR}pd_code_desc_map.csv", index=False))


def calculate_stats(df: pd.DataFrame, columns: List[str]) -> None:
    stats: Dict[str, int] = {}
    for column in columns:
        stats[column] = df[column].isna().sum().astype(str)

    with open(f"{RESULTS_DIR}missing_values.json", "w") as file:
        json.dump(stats, file)


def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-f", "--filepath", required=True, type=str, help="Filepath to CSV file"
    )

    return arg_parser.parse_args()


def create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def check_types() -> None:
    subprocess.call(["mypy", "."])


def check_if_exists_in_args(arg: str) -> bool:
    return arg in sys.argv


def display_finish() -> None:
    print("------------------------------------------------------------------------")
    print("FINISHED")
    print("------------------------------------------------------------------------")


class IdentifierLabels:
    ID = "CMPLNT_NUM"


class DateTimeEventLabels:
    EVENT_START_DATE = "CMPLNT_FR_DT"
    EVENT_START_TIME = "CMPLNT_FR_TM"
    EVENT_START_TIMESTAMP = "CMPLNT_FR_TIMESTAMP"
    EVENT_END_DATE = "CMPLNT_TO_DT"
    EVENT_END_TIME = "CMPLNT_TO_TM"
    EVENT_END_TIMESTAMP = "CMPLNT_TO_TIMESTAMP"


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


if __name__ == "__main__":
    if check_if_exists_in_args("-t"):
        check_types()
    else:
        main()
