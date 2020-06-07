from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple, Union

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


def create_confusion_matrix(
    samples: numpy.ndarray, data: NumpySequences,
) -> numpy.ndarray:

    euclideans: List[List[List[float]]] = [
        [[] for _ in range(len(samples))] for _ in range(len(samples))
    ]

    for sequences, labels in data:
        labels = labels.argmax(axis=-1)
        for sample_label, sample in enumerate(samples):
            diffs = sequences - sample
            euclidean = numpy.linalg.norm(diffs, axis=(-1, -2))
            for i, data_label in enumerate(labels):
                if len(samples) > 1:
                    euclideans[sample_label][data_label].append(euclidean[i])
                else:
                    euclideans[0][0].append(euclidean[i])

    for i in range(len(euclideans)):
        for j in range(len(euclideans[i])):
            lst = euclideans[i][j]
            euclideans[i][j] = min(lst)  # type: ignore

    return numpy.array(euclideans)


def create_epoch_measurements(
    samples: numpy.ndarray, data: NumpySequences,
) -> numpy.ndarray:

    return create_confusion_matrix(samples, data).min(axis=-1).mean()


def measure(
    samples: Union[numpy.ndarray, Sequence[numpy.ndarray]],
    labels: Union[numpy.ndarray, Sequence[numpy.ndarray]],
    num_classes: int,
    data: NumpySequences,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    euclideans: Dict[int, List[float]] = {i: [] for i in range(num_classes)}
    manhattans: Dict[int, List[float]] = {i: [] for i in range(num_classes)}
    for sequences, labels in data:
        for sample in samples:
            diffs = sequences - sample
            manhattan = abs(diffs).sum(axis=(-1, -2))
            euclidean = numpy.linalg.norm(diffs, axis=(-1, -2))
            for i, label in enumerate(labels):
                manhattans[label].append(manhattan[i])
                euclideans[label].append(euclidean[i])
    return (
        {k: min(v) for k, v in euclideans.items()},
        {k: min(v) for k, v in manhattans.items()},
    )
