import subprocess
import sys
from argparse import ArgumentParser, Namespace

import pandas as pd

"""
How to run:
    python dataset_preprocessing.py -f ../data/NYPD_Complaint_Data_Historic.csv -o ../data/NYPD_Data_Preprocessed.csv
"""


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


def main() -> None:
    args = prepare_args()
    filepath = args.filepath
    output = args.output

    print("Loading data...")
    df: pd.DataFrame = pd.read_csv(filepath)
    print("Size of loaded data:", len(df.index))

    df[DateTimeEventLabels.EVENT_START_TIMESTAMP] = (
        (df[DateTimeEventLabels.EVENT_START_DATE] + df[DateTimeEventLabels.EVENT_START_TIME])
            .apply(pd.to_datetime, format='%m/%d/%Y%H:%M:%S', errors='coerce')
    )
    df = df.drop(columns=[DateTimeEventLabels.EVENT_START_DATE, DateTimeEventLabels.EVENT_START_TIME])

    df[DateTimeEventLabels.EVENT_END_TIMESTAMP] = (
        (df[DateTimeEventLabels.EVENT_END_DATE] + df[DateTimeEventLabels.EVENT_END_TIME])
            .apply(pd.to_datetime, format='%m/%d/%Y%H:%M:%S', errors='coerce')
    )
    df = df.drop(columns=[DateTimeEventLabels.EVENT_END_DATE, DateTimeEventLabels.EVENT_END_TIME])

    df[DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_TIMESTAMP] = pd.to_datetime(
        df[DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_DATE]
    )
    df = df.drop(columns=[DateTimeSubmissionLabels.SUBMISSION_TO_POLICE_DATE])

    print("Saving data to file...")
    df.to_csv(output, index=False)

    display_finish()


def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-f", "--filepath", required=True, type=str, help="Filepath to CSV file"
    )
    arg_parser.add_argument(
        "-o", "--output", required=True, type=str, help="Filename of output file"
    )

    return arg_parser.parse_args()


def check_types() -> None:
    subprocess.call(["mypy", "."])


def check_if_exists_in_args(arg: str) -> bool:
    return arg in sys.argv


def display_finish() -> None:
    print("------------------------------------------------------------------------")
    print("FINISHED")
    print("------------------------------------------------------------------------")


if __name__ == "__main__":
    if check_if_exists_in_args("-t"):
        check_types()

    main()
