import logging
from typing import Any

LOGS_FILENAME: str = "app.log"


class Logger:

    def __init__(self) -> None:
        logging.basicConfig(
            filename=LOGS_FILENAME,
            level=logging.INFO,
            format="[%(asctime)s] {%(pathname)s:%(lineno)d} - %(message)s"
        )


    def get_logging_instance(self) -> Any:
        return logging
