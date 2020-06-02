from random import seed, shuffle
from typing import Any, Optional, Sequence, Tuple

import numpy
from keras.utils import Sequence as KerasSequence
from keras.utils import to_categorical
from pandas import DataFrame, Series


class WindowSequence(Sequence[Tuple[DataFrame, DataFrame]]):
    def __init__(
        self,
        sequences: Sequence[DataFrame],
        activity_column: str,
        window: int,
        stride: Optional[int] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.sequences = sequences
        self.activity_column = activity_column
        self.window = window
        self.stride = stride if stride else 1
        self.shuffle = shuffle
        if shuffle:
            self.random_indexes = self.create_random_indexes(seed)

    def __getitem__(self, key: Any) -> Tuple[DataFrame, Series]:
        if type(key) is not int:
            raise ValueError(f"`key` must be an integer: key = {key}")
        if self.shuffle:
            key = self.random_indexes[key]
        i = 0
        len_df = self.len_of_dataframe(self.sequences[i])
        while key > len_df:
            key -= len_df
            i += 1
            len_df = self.len_of_dataframe(self.sequences[i])
        df = (
            self.sequences[i]
            .iloc[key * self.stride : key * self.stride + self.window]
            .reset_index(drop=True)
        )
        return (df.drop(columns=[self.activity_column]), df[self.activity_column])

    def __len__(self) -> int:
        return sum(self.len_of_dataframe(df) for df in self.sequences)

    def get_shape(self, only_numeric: bool = True) -> Tuple[int, int]:
        num_columns = None
        if not only_numeric:
            num_columns = len(self.sequences[0].columns)
        else:
            num_columns = len(self.sequences[0].select_dtypes("number").columns)

        return (self.window, num_columns)

    def create_random_indexes(self, seed_value: Optional[int]) -> Tuple[int, ...]:
        indexes = list(range(len(self)))
        if seed_value:
            seed(seed)
        shuffle(indexes)
        return tuple(indexes)

    def len_of_dataframe(self, df: DataFrame) -> int:
        return (len(df) - self.window) // self.stride

    def to_keras_sequence(self, batch_size: int) -> KerasSequence:
        return NumpySequences(self, batch_size)


class NumpySequences(KerasSequence):
    def __init__(self, window_sequence: WindowSequence, batch_size: int):
        self.window_sequence = window_sequence
        self.shape = (
            len(window_sequence),
            batch_size,
            *window_sequence.get_shape(only_numeric=True),
        )
        self.batch_size = batch_size
        self.activity_codes = {
            a: i
            for i, a in enumerate(sorted(self.window_sequence[0][1].cat.categories))
        }

    def __getitem__(self, index: int) -> numpy.ndarray:
        x = None
        y = None

        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            window = self.window_sequence[i]
            activities = window[1]
            x_next = numpy.expand_dims(window[0].select_dtypes("number").values, axis=0)
            x = numpy.concatenate((x, x_next)) if x is not None else x_next  # type: ignore
            y_next = numpy.expand_dims(
                to_categorical(
                    activities.apply(self.activity_codes.get).values,
                    len(self.activity_codes),
                ),
                axis=0,
            )
            y = numpy.concatenate((y, y_next)) if y is not None else y_next  # type: ignore

        return x, y

    def __len__(self) -> int:
        return len(self.window_sequence) // self.batch_size
