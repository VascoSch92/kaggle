import time
import collections
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from tools.load import load_config
from tools.logger import LoggerSetup
from tools.schema import Schema


class Task(ABC, LoggerSetup):
    config_path: str = None

    def __init__(self) -> None:
        super().__init__()
        if self.config_path:
            self.config = load_config(filepath=Path(self.config_path))

    def skip_task(self) -> bool:
        """Overwrite this method in a subclass if you want to skip the task."""
        return False

    def run(self) -> None:
        if self.skip_task():
            self.logger.info(f"Task {self.__class__.__name__} skipped.")
        else:
            self.logger.info(f"Starting task `{self.__class__.__name__}`")
            start_time = time.time()
            self.run_task()
            end_time = time.time()
            self.logger.info(f"Task completed in {round(end_time - start_time, 2)} seconds")

    @abstractmethod
    def run_task(self) -> None:
        raise NotImplementedError


@dataclass
class Data:
    train: pd.DataFrame
    test: pd.DataFrame
    schema: Optional[Schema] = None


class Pipeline(LoggerSetup):
    def __init__(self, *args: Any) -> None:
        super().__init__()
        self.tasks = self._validate_tasks(args)

    def _validate_tasks(self, args) -> List[Task]:
        for task in args:
            if not issubclass(task, Task):
                raise TypeError(f"Object {task.__repr__()} must be a subclass of `Task`.")
        return list(args)

    def run(self) -> None:
        tasks_names = self._get_tasks_names()
        times = collections.defaultdict(float)
        self.logger.info(f"Starting pipeline: {' -> '.join(tasks_names)}")
        for task_name, task in zip(tasks_names, self.tasks):
            start_time = time.time()
            task().run()
            end_time = time.time()
            times[task_name] = round(end_time - start_time, 2)
        self.logger.info(self._get_analysis_pipeline(times=times))

    def _get_tasks_names(self) -> List[str]:
        return [str(task).split(".")[-1][:-2] for task in self.tasks]

    def _get_analysis_pipeline(self, times: Dict[str, float]) -> str:
        msg = ""
        msg += f"Pipeline completed in {sum(times.values())} seconds. \n"
        msg += "Details:"
        for k, v in times.items():
            msg += "\n"
            msg += f" - {k}: {v} seconds"
        return msg
