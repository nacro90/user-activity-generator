import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Literal, TypeVar

from pandas import DataFrame, read_parquet
from pyarrow import Table, parquet

from .dataset import Dataset
from .filetype import FileType


class DataManager:

    INTERIM_ROOT: ClassVar[Path] = Path(os.environ["INTERIM_ROOT"])

    HASH_LENGTH: ClassVar[int] = int(os.environ["HASH_LENGTH"])

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def convert_dataset(self) -> None:
        print(f"Converting raw to interim: {self.path}")
        self.dataset.read().to_parquet(self.path)

    def read(self) -> DataFrame:
        return read_parquet(self.path)

    def read_meta(self) -> DataFrame:
        if self.need_raw():
            self.convert_dataset()
        # TODO

    def need_raw(self) -> bool:
        if self.path.exists() and not self.has_missing_features():
            print("Interim exists")
            return False
        return True

    def has_missing_features(self) -> Literal[False]:
        return False
        #  self.dataset.generators.keys()

    @property
    def path(self) -> Path:
        return Path(
            DataManager.INTERIM_ROOT / self.dataset.hash[: DataManager.HASH_LENGTH]
        )
