import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, List, Literal, Optional, Sequence, Set, Tuple

import pandas
import pyarrow
from pandas import DataFrame
from pyarrow.parquet import ParquetFile, ParquetSchema

from ..config import Config
from .dataset import Activity, Dataset
from .filetype import FileType


class Windows(Sequence[DataFrame]):
    def __init__(
        self, dataframes: Sequence[DataFrame], window: int, stride: Optional[int] = None
    ) -> None:
        self.dataframes = dataframes
        self.window = window
        self.stride = stride if stride else window

    def __getitem__(self, key: Any) -> DataFrame:
        if type(key) is not int:
            raise ValueError(f"`key` must be an integer: key = {key}")
        i = 0
        len_df = self.len_of_dataframe(self.dataframes[i])
        while key > len_df:
            key -= len_df
            i += 1
            len_df = self.len_of_dataframe(self.dataframes[i])
        return (
            self.dataframes[i]
            .iloc[key * self.stride : key * self.stride + self.window]
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        return sum(self.len_of_dataframe(df) for df in self.dataframes)

    def len_of_dataframe(self, df: DataFrame) -> int:
        return (len(df) - self.window) // self.stride


class DataManager:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def generate_features(self, raw: DataFrame) -> None:
        generators = self.dataset.generators()
        if generators:
            for gen_key, gen_function in generators.items():
                raw[gen_key] = gen_function(raw)

    def convert_dataset(self) -> DataFrame:
        print(f"Converting raw to interim: {self.path}")
        self.delete_interim()
        raw_dataframe = self.dataset.read()
        self.generate_features(raw_dataframe)
        self.path.parent.mkdir(0o755, parents=True, exist_ok=True)
        raw_dataframe.to_parquet(self.path)
        return raw_dataframe

    def read(self, columns: List[str] = None, clean: bool = False) -> DataFrame:
        if clean or self.is_interim_dirty():
            self.convert_dataset()
        return pandas.read_parquet(
            self.path, columns=columns if columns else list(self.dataset.COLUMNS.keys())
        )

    def stream(
        self,
        activities: Set[Activity],
        window: int,
        stride: Optional[int] = None,
        subjects: Optional[Set[int]] = None,
        clean: bool = False,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> DataFrame:
        stride = stride if stride else window
        data = self.read(clean=clean)
        activity_keys = [self.dataset.ACTIVITIES[activity] for activity in activities]
        filtered = data.loc[data[self.dataset.ACTIVITY_COLUMN].isin(activity_keys)]
        if subjects:
            filtered = filtered.loc[
                filtered[self.dataset.SUBJECT_COLUMN].isin(subjects)
            ]
        dataframes = [
            data
            for _, data in filtered.groupby(
                [self.dataset.TRIAL_COLUMN, self.dataset.SUBJECT_COLUMN]
            )
        ]
        if shuffle:
            if seed:
                random.seed(seed)
            random.shuffle(dataframes)
        return Windows(dataframes, window, stride)

    def read_schema(self) -> ParquetSchema:
        return ParquetFile(self.path).schema

    def is_interim_dirty(self) -> bool:
        return not self.path.exists() or self.any_missing_features()

    def any_missing_features(self) -> bool:
        interim_cols = self.read_schema().names
        raw_cols = self.dataset.COLUMNS.keys()
        return any([col not in interim_cols for col in raw_cols])

    def delete_interim(self) -> None:
        if self.path.exists():
            self.path.unlink()

    @property
    def path(self) -> Path:
        return Path(Config.INTERIM_ROOT / self.dataset.hash[: Config.HASH_LENGTH])
