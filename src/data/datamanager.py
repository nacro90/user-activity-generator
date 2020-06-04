import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, List, Literal, Optional, Sequence, Set, Tuple

import numpy
import pandas
import pyarrow
from keras.utils import Sequence as KerasSequence
from keras.utils import to_categorical
from pandas import DataFrame
from pyarrow.parquet import ParquetFile, ParquetSchema

from ..config import Config
from .dataset import Activity, Dataset
from .filetype import FileType
from .window import KerasSequence, WindowSequence


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

    def create_windows(
        self,
        activities: Set[Activity],
        window: int,
        stride: Optional[int] = None,
        subjects: Optional[Set[int]] = None,
        clean: bool = False,
        shuffle: bool = False,
        seed: Optional[int] = None,
        columns: List[str] = None,
    ) -> DataFrame:
        stride = stride if stride else window
        data = self.read(
            clean=clean,
            columns=(
                columns
                + [
                    self.dataset.ACTIVITY_COLUMN,
                    self.dataset.TRIAL_COLUMN,
                    self.dataset.SUBJECT_COLUMN,
                ]
            )
            if columns
            else None,
        )
        activity_keys = [self.dataset.ACTIVITIES[activity] for activity in activities]
        filtered = data.loc[data[self.dataset.ACTIVITY_COLUMN].isin(activity_keys)]
        if subjects:
            filtered = filtered.loc[
                filtered[self.dataset.SUBJECT_COLUMN].isin(subjects)
            ]
        dataframes = [
            data.drop(columns=[self.dataset.SUBJECT_COLUMN, self.dataset.TRIAL_COLUMN])
            for _, data in filtered.groupby(
                [self.dataset.TRIAL_COLUMN, self.dataset.SUBJECT_COLUMN]
            )
        ]
        return WindowSequence(
            dataframes,
            self.dataset.ACTIVITY_COLUMN,
            len(activities),
            window,
            stride=stride,
            shuffle=shuffle,
            seed=seed,
        )

    def df_to_np(self, dataframe: DataFrame) -> numpy.ndarray:
        activity_enumeration = self.dataset.enumerate_activities()
        activity_codes = dataframe[self.dataset.ACTIVITY_COLUMN].apply(
            activity_enumeration.get
        )
        numerical_attributes = dataframe.select_dtypes("number")
        return (
            numerical_attributes.values,
            to_categorical(
                activity_codes.values, num_classes=len(activity_enumeration)
            ),
        )

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
