from ..config import Config
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar

import pandas
import pyarrow
from pandas import DataFrame
from pyarrow.parquet import ParquetFile, ParquetSchema

from .dataset import Dataset
from .filetype import FileType


class DataManager:

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def generate_features(self, raw: DataFrame) -> None:
        if self.dataset.generators:
            for gen_key, gen_function in self.dataset.generators.items():
                raw.insert(len(raw.columns), gen_key, gen_function(raw))

    def convert_dataset(self) -> DataFrame:
        print(f"Converting raw to interim: {self.path}")
        self.delete_interim()
        raw_dataframe = self.dataset.read()
        self.generate_features(raw_dataframe)
        raw_dataframe.to_parquet(self.path)
        return raw_dataframe

    def read(self, clean: bool = False) -> DataFrame:
        if clean or self.is_interim_dirty():
            self.convert_dataset()
        return pandas.read_parquet(self.path, columns=self.dataset.columns)

    def get_data(self) -> Any:
        pass # TODO

    def read_schema(self) -> ParquetSchema:
        return ParquetFile(self.path).schema

    def is_interim_dirty(self) -> bool:
        return not self.path.exists() or self.any_missing_features()

    def any_missing_features(self) -> bool:
        interim_cols = self.read_schema().names
        raw_cols = self.dataset.columns.keys()
        return any([col not in interim_cols for col in raw_cols])

    def delete_interim(self) -> None:
        self.path.unlink(True)

    @property
    def path(self) -> Path:
        return Path(
            Config.INTERIM_ROOT / self.dataset.hash[: Config.HASH_LENGTH]
        )
