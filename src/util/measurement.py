from typing import List, Optional, Sequence, Tuple, Union

import dtw
import numpy

from ..data.window import NumpySequences


def min_euclidean(
    samples: Union[numpy.ndarray, Sequence[numpy.ndarray]], data: NumpySequences,
) -> float:
    distances: List[float] = []
    for sample in samples:
        min_dist: Optional[float] = None
        for sequences, _ in data:
            for sequence in sequences:
                dist = numpy.linalg.norm(sequence - sample)
                min_dist = min(min_dist, dist) if min_dist else dist  # type: ignore
        distances.append(min_dist if min_dist else 0)
    return sum(distances) / len(distances)


def dynamic_time_warp(
    samples: Union[numpy.ndarray, Sequence[numpy.ndarray]], data: NumpySequences,
) -> float:
    distances: List[float] = []
    for sample in samples:
        min_dist: Optional[float] = None
        for sequences, _ in data:
            for sequence in sequences:
                dist = dtw.accelerated_dtw(
                    sample, sequence, lambda a, b: numpy.linalg.norm(a - b)
                )
                min_dist = min(min_dist, dist) if min_dist else dist  # type: ignore
        distances.append(min_dist if min_dist else 0)
    return sum(distances) / len(distances)


def measure(
    samples: Union[numpy.ndarray, Sequence[numpy.ndarray]], data: NumpySequences
) -> Tuple[float, float]:
    euclideans: List[float] = []
    manhattans: List[float] = []
    for sample in samples:
        min_euclidean: Optional[float] = None
        min_manhattan: Optional[float] = None
        for sequences, _ in data:
            for sequence in sequences:
                diff = sequence - sample
                manhattan = abs(diff).sum()
                min_manhattan = (
                    min(min_manhattan, manhattan) if min_manhattan else manhattan  # type: ignore
                )
                euclidean = numpy.linalg.norm(diff)
                min_euclidean = (
                    min(min_euclidean, euclidean) if min_euclidean else euclidean  # type: ignore
                )
        euclideans.append(min_euclidean if min_euclidean else 0)
        manhattans.append(min_manhattan if min_manhattan else 0)
    return sum(euclideans) / len(euclideans), sum(manhattans) / len(manhattans)  # type: ignore
