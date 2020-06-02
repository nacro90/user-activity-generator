from typing import Sequence, Union

import dtw
import numpy

from ..data.window import NumpySequences


def min_euclidean(
    samples: Union[numpy.ndarray, Sequence[numpy.ndarray]], data: NumpySequences,
) -> float:
    distances = []
    for sample in samples:
        min_dist = None
        for sequences, _ in data:
            for sequence in sequences:
                dist = numpy.linalg.norm(sequence - sample)
                min_dist = min(min_dist, dist) if min_dist else dist
        distances.append(min_dist)
    return sum(distances) / len(distances)


def dynamic_time_warp(
    samples: Union[numpy.ndarray, Sequence[numpy.ndarray]], data: NumpySequences,
) -> float:
    distances = []
    for sample in samples:
        min_dist = None
        for sequences, _ in data:
            for sequence in sequences:
                dist = dtw.accelerated_dtw(
                    sample, sequence, lambda a, b: numpy.linalg.norm(a - b)
                )
                min_dist = min(min_dist, dist) if min_dist else dist
        distances.append(min_dist)
    return sum(distances) / len(distances)


def measure(
    samples: Union[numpy.ndarray, Sequence[numpy.ndarray]], data: NumpySequences
) -> float:
    euclideans = []
    manhattans = []
    for sample in samples:
        min_euclidean = None
        min_manhattan = None
        for sequences, _ in data:
            for sequence in sequences:
                diff = sequence - sample
                manhattan = abs(diff).sum()
                min_manhattan = (
                    min(min_manhattan, manhattan) if min_manhattan else manhattan
                )
                euclidean = numpy.linalg.norm(diff)
                min_euclidean = (
                    min(min_euclidean, euclidean) if min_euclidean else euclidean
                )
        euclideans.append(min_euclidean)
        manhattans.append(min_manhattan)
    return sum(euclideans) / len(euclideans), sum(manhattans) / len(manhattans)
