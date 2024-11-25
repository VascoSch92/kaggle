from typing import Dict, List, Union
from pathlib import Path

import yaml


def _dict_to_class(class_name, dict_obj):
    class_attrs = {}

    for key, value in dict_obj.items():
        if isinstance(value, dict):
            class_attrs[key] = _dict_to_class(key.capitalize(), value)
        else:
            class_attrs[key] = value
    return type(class_name, (object,), class_attrs)


def _load_yaml_file(filepath: Path) -> Union[Dict, List[Dict]]:
    with Path(filepath).open("r") as file:
        cfg = yaml.safe_load(file)
    return cfg


def load_config(filepath: Path) -> "Params":
    cfg = _load_yaml_file(filepath=filepath)
    return _dict_to_class("Params", cfg)
