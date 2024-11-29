from uuid import uuid4
from typing import Union
from logging import Logger
from pathlib import Path

import pandas as pd


def save_submission_as_csv(path: Union[Path, str], submission: pd.DataFrame, logger: Logger) -> None:
    submission_identifier = str(uuid4())[:8]
    submission.to_csv(Path(path) / f"submission_{submission_identifier}.csv", index=False)
    logger.info(f"Submission {submission_identifier} successfully saved in {path}!")
