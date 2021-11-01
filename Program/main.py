from argparse import ArgumentParser, Namespace

import pandas as pd

from module.Logger import Logger
from module.utils import check_if_exists_in_args, check_types_check_style, compile_to_pyc, \
    display_finish

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
    if check_if_exists_in_args("-t"):
        check_types_check_style()
    elif check_if_exists_in_args("-b"):
        compile_to_pyc()
    else:
        main()
