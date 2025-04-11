import pandas as pd
import numpy as np
from itertools import permutations

def calculate_permutation_entropy(time_series, m, delay):
    """
    Calculate the permutation entropy of a given time series.
    """
    n = len(time_series)
    permutations_set = set()

    for i in range(n - m * delay):
        pattern = tuple(time_series[i + j * delay] for j in range(m))
        permutations_set.add(pattern)

    permutations_count = len(permutations_set)
    return -np.log2(permutations_count / (n - m * delay + 1))


def _algorithm(time_series, max_m, delay=1):
    """
    Estimate the minimum embedding dimension using _algorithm.
    time_series: time series data
    max_m: Maximum embedding dimensions
    delay: delay time
    """
    entropy_values = []
    for m in range(1, max_m + 1):
        entropy = calculate_permutation_entropy(time_series, m, delay)
        entropy_values.append(entropy)
        # print(f"m={m}, Permutation Entropy={entropy}")


    min_embedding_dim = None
    for i in range(1, len(entropy_values)):
        if entropy_values[i] >= entropy_values[i - 1]:
            min_embedding_dim = i
            break

    if min_embedding_dim is None:
        min_embedding_dim = max_m

    return min_embedding_dim
