from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple, Union

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


def create_confusion_matrix(
    samples: numpy.ndarray, data: NumpySequences, num_classes: int
) -> numpy.ndarray:

    euclideans: List[List[List[float]]] = [
        [[] for _ in range(num_classes)] for _ in range(len(samples))
    ]

    for sequences, labels in data:
        labels = labels.argmax(axis=-1)
        for sample_label, sample_set in enumerate(samples):
            diff_list = [sequences - sample for sample in sample_set]
            diffs = numpy.asarray(diff_list)
            diffs = diffs.mean(axis=0)
            euclidean = numpy.linalg.norm(diffs, axis=(-1, -2))
            for i, data_label in enumerate(labels):
                euclideans[sample_label][data_label].append(euclidean[i])

    for i in range(len(euclideans)):
        for j in range(len(euclideans[i])):
            lst = euclideans[i][j]
            euclideans[i][j] = min(lst)  # type: ignore

    return numpy.array(euclideans)


def create_epoch_measurements(
    samples: numpy.ndarray, data: NumpySequences, num_classes: int
) -> numpy.ndarray:
    return create_confusion_matrix(samples, data, num_classes).min(axis=-1).mean()


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
