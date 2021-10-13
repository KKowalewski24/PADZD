import subprocess
import sys
from argparse import ArgumentParser, Namespace

import pandas as pd
from module.Logger import Logger

"""
    How to run
        python main.py
        
    Remember to place file with data in 'data' directory!
"""

DATA_FILEPATH = "data/NYPD_Data_Trimmed.csv"


def main() -> None:
    logger = Logger().get_logging_instance()
    args = prepare_args()
    logger.info("Start program with args: " + str(vars(args)))
    df = pd.read_csv(DATA_FILEPATH)

    display_finish()


def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    return arg_parser.parse_args()


def check_types_check_style() -> None:
    subprocess.call(["mypy", "."])
    subprocess.call(["flake8", "."])


def compile_to_pyc() -> None:
    subprocess.call(["python", "-m", "compileall", "."])


def check_if_exists_in_args(arg: str) -> bool:
    return arg in sys.argv


def display_finish() -> None:
    print("------------------------------------------------------------------------")
    print("FINISHED")
    print("------------------------------------------------------------------------")


if __name__ == "__main__":
    if check_if_exists_in_args("-t"):
        check_types_check_style()
    elif check_if_exists_in_args("-b"):
        compile_to_pyc()
    else:
        main()
