from abc import ABC, abstractmethod
from enum import Enum
from math import sqrt
from pathlib import Path
from typing import Callable, Dict, Optional, Type, ClassVar, Tuple

import pandas
from pandas import DataFrame, Series

from ..util.integrity import recursive_sha256
from .filetype import Csv, FileType
from .reader import CsvReader, Reader


class Activity(Enum):
    WALKING = 0
    JOGGING = 1
    UPSTAIRS = 2
    DOWNSTAIRS = 3
    SITTING = 4
    STANDING = 5


class Dataset(ABC):
    ACTIVITY_COLUMN: ClassVar[str] = NotImplemented
    ACTIVITIES: ClassVar[Dict[Activity, str]] = NotImplemented
    COLUMNS: ClassVar[Dict[str, Reader.DataType]] = NotImplemented
    FREQUENCY: ClassVar[int] = NotImplemented
    TRIAL_COLUMN: ClassVar[str] = NotImplemented
    SUBJECT_COLUMN: ClassVar[str] = NotImplemented

    @classmethod
    @abstractmethod
    def generators(cls) -> Optional[Dict[str, Callable[[DataFrame], Series]]]:
        raise NotImplementedError

    @classmethod
    def is_columns_valid(cls) -> bool:
        generators = cls.generators()
        if not generators:
            return True
        return all(gen_key in cls.COLUMNS.keys() for gen_key in generators.keys())

    def __init__(self, path: Path):
        if not self.is_columns_valid():
            raise ValueError("All generator keys must be specified in column field")
        self.path = path

    @property
    def hash(self) -> str:
        return recursive_sha256(self.path)

    @abstractmethod
    def read(self) -> DataFrame:
        pass


class Wisdm(Dataset):

    ACTIVITIES = {
        Activity.WALKING: "Walking",
        Activity.JOGGING: "Jogging",
        Activity.UPSTAIRS: "Upstairs",
        Activity.DOWNSTAIRS: "Downstairs",
        Activity.SITTING: "Sitting",
        Activity.STANDING: "Standing",
    }

    ACTIVITY_COLUMN = "activity"
    TRIAL_COLUMN = "trial"
    SUBJECT_COLUMN = "subject"

    COLUMNS = {
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

    FREQUENCY = 20

    @classmethod
    def generators(cls) -> Optional[Dict[str, Callable[[DataFrame], Series]]]:
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

    def __init__(self, path: Path) -> None:
        Dataset.__init__(self, path)

    def read(self) -> DataFrame:
        reader = CsvReader(self.path)
        return reader.read(self.COLUMNS)


class MotionSense(Dataset):

    ACTIVITIES = {
        Activity.WALKING: "wlk",
        Activity.JOGGING: "jog",
        Activity.UPSTAIRS: "ups",
        Activity.DOWNSTAIRS: "dws",
        Activity.SITTING: "sit",
        Activity.STANDING: "std",
    }

    COLUMNS = {
        "subject": Reader.DataType.INT64,
        "trial": Reader.DataType.INT64,
        "activity": Reader.DataType.CATEGORY,
        "attitude.roll": Reader.DataType.FLOAT64,
        "attitude.pitch": Reader.DataType.FLOAT64,
        "attitude.yaw": Reader.DataType.FLOAT64,
        "gravity.x": Reader.DataType.FLOAT64,
        "gravity.y": Reader.DataType.FLOAT64,
        "gravity.z": Reader.DataType.FLOAT64,
        "rotationRate.x": Reader.DataType.FLOAT64,
        "rotationRate.y": Reader.DataType.FLOAT64,
        "rotationRate.z": Reader.DataType.FLOAT64,
        "userAcceleration.x": Reader.DataType.FLOAT64,
        "userAcceleration.y": Reader.DataType.FLOAT64,
        "userAcceleration.z": Reader.DataType.FLOAT64,
        "magnitude": Reader.DataType.FLOAT64,
        "xaccel_norm": Reader.DataType.FLOAT64,
        "yaccel_norm": Reader.DataType.FLOAT64,
        "zaccel_norm": Reader.DataType.FLOAT64,
        "magnitude_norm": Reader.DataType.FLOAT64,
    }

    FREQUENCY = 50

    ACTIVITY_COLUMN = "activity"

    SUBJECT_COLUMN = "subject"
    TRIAL_COLUMN = "trial"

    @classmethod
    def generators(cls) -> Dict[str, Callable[[DataFrame], Series]]:
        def magnitude(df: DataFrame) -> Series:
            xacc = df["userAcceleration.x"]
            yacc = df["userAcceleration.y"]
            zacc = df["userAcceleration.z"]
            euclidean = (xacc ** 2 + yacc ** 2 + zacc ** 2) ** 0.5
            return Series(euclidean)

        def normalize(series: Series) -> Series:
            return Series((series - series.mean()) / (series.max() - series.min()))

        return {
            "magnitude": magnitude,
            "xaccel_norm": lambda df: normalize(df["userAcceleration.x"]),
            "yaccel_norm": lambda df: normalize(df["userAcceleration.y"]),
            "zaccel_norm": lambda df: normalize(df["userAcceleration.z"]),
            "magnitude_norm": lambda df: normalize(magnitude(df)),
        }

    def __init__(self, path: Path) -> None:
        Dataset.__init__(self, path)

    def read(self) -> DataFrame:
        pandas_columns = {
            name: type_enum.value for name, type_enum in self.COLUMNS.items()
        }
        concated = DataFrame(columns=pandas_columns)
        for folder in self.path.iterdir():
            activity, trial = self.split_activity_and_trial(folder.name)
            for file in folder.iterdir():
                df = CsvReader(file).read(self.COLUMNS)
                df.drop(columns="Unnamed: 0", inplace=True)
                df["subject"] = self.strip_subject_no(file.name)
                df["trial"] = trial
                df["activity"] = activity
                concated = pandas.concat((concated, df))

        return concated.astype(pandas_columns)

    def strip_subject_no(self, fname: str) -> int:
        return int(fname.split("_")[1].split(".")[0])

    def split_activity_and_trial(self, fname: str) -> Tuple[str, int]:
        split = fname.split("_")
        return split[0], int(split[1])

        # reader = CsvReader(self.path)
        # return reader.read(self.COLUMNS)
