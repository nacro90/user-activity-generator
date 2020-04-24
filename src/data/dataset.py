from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional, Type

from pandas import DataFrame, Series

from ..util.integrity import recursive_sha256
from .filetype import Csv, FileType
from .reader import CsvReader, Reader


class Dataset(ABC):
    def __init__(
        self,
        path: Path,
        file_type: Type[FileType],
        columns: Dict[str, Reader.DataType],
        generators: Optional[Dict[str, Callable[[DataFrame], Series]]],
    ):
        self.path = path
        self.columns = columns
        self.file_type = file_type
        self.generators = generators

    @property
    def hash(self) -> str:
        return recursive_sha256(self.path)

    @abstractmethod
    def read(self) -> DataFrame:
        pass


class Wisdm(Dataset):

    USERS = set(range(1, 37))

    def __init__(self, path: Path) -> None:
        columns = {
            "user": Reader.DataType.CATEGORY,
            "activity": Reader.DataType.CATEGORY,
            "timestamp": Reader.DataType.INT64,
            "xaccel": Reader.DataType.FLOAT64,
            "yaccel": Reader.DataType.FLOAT64,
            "zaccel": Reader.DataType.FLOAT64,
        }
        Dataset.__init__(self, path, Csv, columns, None)

    def read(self) -> DataFrame:
        reader = CsvReader(self.path)
        return reader.read(self.columns)

    class Activity(Enum):
        WALKING = "Walking"
        JOGGING = "Jogging"
        UPSTAIRS = "Upstairs"
        DOWNSTAIRS = "Downstairs"
        SITTING = "Sitting"
        STANDING = "Standing"
