import os
import subprocess
import sys
from datetime import datetime
from typing import Callable


def create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def prepare_filename(name: str, extension: str = "", add_date: bool = True) -> str:
    return (name + ("-" + datetime.now().strftime("%H%M%S") if add_date else "")
            + extension).replace(" ", "")


def run_main(main: Callable[[], None]) -> None:
    if check_if_exists_in_args("-t"):
        check_types_check_style()
    elif check_if_exists_in_args("-b"):
        compile_to_pyc()
    else:
        main()


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
