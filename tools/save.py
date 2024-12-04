import json
from uuid import uuid4
from typing import Dict, Union
from logging import Logger
from pathlib import Path

import pandas as pd


def save_submission_as_csv(path: Union[Path, str], submission: pd.DataFrame, logger: Logger) -> None:
    submission_identifier = str(uuid4())[:8]
    submission.to_csv(Path(path) / f"submission_{submission_identifier}.csv", index=False)
    logger.info(f"Submission {submission_identifier} successfully saved in {path}!")


def save_parameters(path: Union[Path, str], parameters: Dict, logger: Logger) -> None:
    with Path(path).open("w") as file:
        json.dump(parameters, file)
    logger.info(f"Parameters successfully saved in {path}!")
