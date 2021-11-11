from argparse import ArgumentParser, Namespace

import pandas as pd

from module.Logger import Logger
from module.utils import display_finish, run_main

"""
    How to run
        python main.py
        
    Remember to place file with data in 'data' directory!
"""

DATA_FILEPATH = "data/NYPD_Complaint_Data_Historic.csv"


def main() -> None:
    logger = Logger().get_logging_instance()
    args = prepare_args()
    logger.info("Start program with args: " + str(vars(args)))
    save_data = args.save

    logger.info("Loading dataset...")
    df = pd.read_csv(DATA_FILEPATH)
    logger.info("Loading dataset finished")

    display_finish()


def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-s", "--save", default=False, action="store_true", help="Save generated data"
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    run_main(main)
