from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Union

from pandas import DataFrame, read_csv

from ..util.integrity import recursive_sha256
from .filetype import FileType


class Reader(ABC):
    def __init__(self, path: Path):
        if not path.is_file():
            raise ValueError(f"Unsupported path (Not a file): {path}")
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        self.path = path

    @abstractmethod
    def read(self, columns: Dict[str, "DataType"]) -> DataFrame:
        pass

    @property
    def hash(self) -> str:
        return recursive_sha256(self.path)

    class DataType(Enum):
        INT64 = "int64"
        FLOAT64 = "float64"
        BOOL = "bool"
        DATETIME64 = "datetime64"
        OBJECT = "object"
        CATEGORY = "category"
        TIMEDELTA = "timedelta"


class CsvReader(Reader):
    def __init__(self, path: Path):
        Reader.__init__(self, path)

    def read(self, columns: Dict[str, Reader.DataType]) -> DataFrame:
        pandas_columns = {name: type_enum.value for name, type_enum in columns.items()}
        df = read_csv(self.path, dtype=pandas_columns)
        for key, datatype in columns.items():
            if key not in df:
                if datatype in {Reader.DataType.FLOAT64, Reader.DataType.INT64}:
                    df[key] = 0
                elif datatype in {Reader.DataType.CATEGORY, Reader.DataType.CATEGORY}:
                    df[key] = ""
                elif datatype == Reader.DataType.BOOL:
                    df[key] = False

        return df.astype(pandas_columns)
