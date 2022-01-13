from argparse import ArgumentParser, Namespace

import pandas as pd

"""
How to run:
    python dataset_stats.py -f ../data/NYPD_Complaint_Data_Historic.csv -o ../data/NYPD_stats.csv 
"""


def main() -> None:
    args = prepare_args()
    filepath = args.filepath
    output = args.output

    print("Loading data...")
    df: pd.DataFrame = pd.read_csv(filepath)
    print("Calculating stats...")
    df.describe(include="all").to_csv(output)

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


def display_finish() -> None:
    print("------------------------------------------------------------------------")
    print("FINISHED")
    print("------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
