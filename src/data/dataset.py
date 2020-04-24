from abc import ABC, abstractmethod
from enum import Enum
from math import sqrt
from pathlib import Path
from typing import Callable, Dict, Optional, Type

from pandas import DataFrame, Series

from ..util.integrity import recursive_sha256
from .filetype import Csv, FileType
from .reader import CsvReader, Reader


class Dataset(ABC):
    def __init__(self, path: Path, file_type: Type[FileType]):
        if not self.is_columns_valid():
            raise ValueError("All generator keys must be specified in column field")
        self.path = path
        self.file_type = file_type

    @property
    def hash(self) -> str:
        return recursive_sha256(self.path)

    def is_columns_valid(self) -> bool:
        if not self.generators:
            return True
        return all(gen_key in self.columns.keys() for gen_key in self.generators.keys())

    @property
    @abstractmethod
    def columns(self) -> Dict[str, Reader.DataType]:
        pass

    @property
    @abstractmethod
    def generators(self) -> Optional[Dict[str, Callable[[DataFrame], Series]]]:
        pass

    @abstractmethod
    def read(self) -> DataFrame:
        pass


class Wisdm(Dataset):

    USERS = set(range(1, 37))

    def __init__(self, path: Path) -> None:

        Dataset.__init__(self, path, Csv)

    def read(self) -> DataFrame:
        reader = CsvReader(self.path)
        return reader.read(self.columns)

    @property
    def columns(self) -> Dict[str, Reader.DataType]:
        return {
            "user": Reader.DataType.CATEGORY,
            "activity": Reader.DataType.CATEGORY,
            "timestamp": Reader.DataType.INT64,
            "xaccel": Reader.DataType.FLOAT64,
            "yaccel": Reader.DataType.FLOAT64,
            "zaccel": Reader.DataType.FLOAT64,
            "magnitude": Reader.DataType.FLOAT64,
            "xaccel_norm": Reader.DataType.FLOAT64,
            "yaccel_norm": Reader.DataType.FLOAT64,
            "zaccel_norm": Reader.DataType.FLOAT64,
            "magnitude_norm": Reader.DataType.FLOAT64,
        }

    @property
    def generators(self) -> Dict[str, Callable[[DataFrame], Series]]:
        def magnitude(df: DataFrame) -> Series:
            xacc = df["xaccel"]
            yacc = df["yaccel"]
            zacc = df["zaccel"]
            euclidean = (xacc ** 2 + yacc ** 2 + zacc ** 2) ** 0.5
            return Series(abs(euclidean - 10))

        def normalize(series: Series) -> Series:
            return Series((series - series.mean()) / (series.max() - series.min()))

        return {
            "magnitude": magnitude,
            "xaccel_norm": lambda df: normalize(df["xaccel"]),
            "yaccel_norm": lambda df: normalize(df["yaccel"]),
            "zaccel_norm": lambda df: normalize(df["zaccel"]),
            "magnitude_norm": lambda df: normalize(magnitude(df)),
        }

    class Activity(Enum):
        WALKING = "Walking"
        JOGGING = "Jogging"
        UPSTAIRS = "Upstairs"
        DOWNSTAIRS = "Downstairs"
        SITTING = "Sitting"
        STANDING = "Standing"
