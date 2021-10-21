import subprocess
import sys
from argparse import ArgumentParser, Namespace

import pandas as pd

"""
How to run:
    python dataset_preprocessing.py -f ../data/NYPD_Complaint_Data_Historic.csv -r 1000000 -o ../data/NYPD_Data_Trimmed.csv
"""


def main() -> None:
    args = prepare_args()
    filepath = args.filepath
    nrows = args.nrows
    output = args.output

    print("Loading data...")
    df: pd.DataFrame = pd.read_csv(filepath)
    print("Size of loaded data:", len(df.index))
    # Consider more fancy imputation than "List wise deletion"
    # print("Imputating data...")
    # df = df.dropna()
    # print("Size of data after imputation:", len(df.index))
    print("Saving data to file...")
    df.head(nrows).to_csv(output, index=False)

    display_finish()


def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-f", "--filepath", required=True, type=str, help="Filepath to CSV file"
    )
    arg_parser.add_argument(
        "-r", "--nrows", required=True, type=int, help="Number of rows inserted to result file"
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
