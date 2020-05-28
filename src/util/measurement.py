import numpy


def min_euclidean(a: numpy.ndarray, b: numpy.ndarray) -> float:
    min_norm = 2 ** 63
    if len(a) < len(b):
        a, b = b, a
    for i in range(len(a) - len(b)):
        a_slice = a[i : i + len(b)]
        norm = numpy.linalg.norm(a_slice - b)
        min_norm = min(min_norm, norm)
    return min_norm
