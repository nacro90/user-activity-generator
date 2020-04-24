from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Type


class FileType(ABC):
    @property
    @classmethod
    @abstractmethod
    def suffix(cls) -> str:
        pass


class Csv(FileType):
    suffix: ClassVar[str] = "csv"


class Parquet(FileType):
    suffix: ClassVar[str] = "parquet"


class Json(FileType):
    suffix: ClassVar[str] = "json"


class Numpy(FileType):
    suffix: ClassVar[str] = "npy"
