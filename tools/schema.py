import pickle
from enum import Enum
from typing import Dict, List, Union
from logging import Logger
from pathlib import Path
from dataclasses import field, dataclass


@dataclass
class Schema:
    meta: List[str] = field(default_factory=list)
    numeric: Dict = field(default_factory=dict)
    catvar: Dict = field(default_factory=dict)
    labels: List[str] = field(default_factory=list)

    def numeric_features(self) -> List[str]:
        return list(self.numeric.keys())

    def catvar_features(self) -> List[str]:
        return list(self.catvar.keys())

    def save_as_pickle(self, filepath: Union[Path, str], logger: Logger) -> None:
        with Path(filepath).open("wb") as f:
            pickle.dump(self, f)
        logger.info(self.__repr__())
        logger.info(f"Schema successfully saved at {filepath}!")


class EncodingType(Enum):
    ORDINAL_ENCODING = "ordinal_encoding"
    LABEL_ENCODING = "label_encoding"
    BINARY_ENCODING = "binary_encoding"
    ONE_HOT_ENCODING = "one_hot_encoding"
    TARGET_ENCODING = "target_encoding"
    FREQUENCY_ENCODING = "frequency_encoding"


class ScalerType(Enum):
    MINMAX_SCALER = "minmax_scaler"
    STANDARD_SCALER = "standard_scaler"
