import pickle
from typing import Dict, List, Union
from logging import Logger
from pathlib import Path

import yaml
import pandas as pd

from tools.schema import Schema


def _dict_to_class(class_name, dict_obj):
    class_attrs = {}

    for key, value in dict_obj.items():
        if isinstance(value, dict):
            class_attrs[key] = _dict_to_class(key.capitalize(), value)
        else:
            class_attrs[key] = value
    return type(class_name, (object,), class_attrs)


def _load_yaml_file(filepath: Path) -> Union[Dict, List[Dict]]:
    with filepath.open("r") as file:
        cfg = yaml.safe_load(file)
    return cfg


def load_config(filepath: Path) -> "Params":
    cfg = _load_yaml_file(filepath=filepath)
    return _dict_to_class("Params", cfg)


def load_from_csv(filepath: Union[Path, str], logger: Logger) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    logger.info(f"Loaded dataframe {filepath}: {len(df)} rows x {len(df.columns)} columns.")
    return df


def load_schema(filepath: Union[Path, str], logger: Logger) -> Schema:
    with Path(filepath).open("rb") as f:
        schema = pickle.load(f)
    logger.info(f"Loaded schema from {filepath}")
    return schema
