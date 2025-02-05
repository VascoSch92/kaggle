from typing import Dict
from pathlib import Path

import yaml


class Config:
    def __init__(self, dictionary: Dict) -> None:
        self._construct(dictionary)

    def _construct(self, dictionary: Dict) -> None:
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    @classmethod
    def from_file(cls, filepath: Path) -> "Config":
        with filepath.open("r") as file:
            cfg = yaml.safe_load(file)
        return cls(cfg)

    def add(self, params: Dict) -> None:
        self._construct(params)

    def validate(self, params: Dict) -> None:
        for key, value in params.items():
            if hasattr(self, key) is False:
                error_msg = f"Entry '{key}' not found in config."
                if isinstance(value, str):
                    error_msg += f" {value}"
                raise KeyError(error_msg)
            if isinstance(value, dict):
                getattr(self, key).validate(value)


if __name__ == "__main__":
    config = Config({})
    config.add({"key": "value"})
    config.add(
        {
            "a": "b",
            "c": {
                "e": 1,
                "f": 2,
            },
        }
    )
    print(config.c.e)
    config.validate(
        {
            "a": "b",
            "c": {
                "e": 1,
                "f": 2,
            },
        }
    )
