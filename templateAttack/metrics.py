"""
Authors : 
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari,
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

import numpy as np


def rankKey(data: np.ndarray,
            key: int) -> int:
    """
    Computes the rank of `key` in `data`, i.e.,
    the position of `key` in the sorted array `data`.

    Parameters
    ----------
    `data` :  array_like
        The data to rank.
    `key`  : int
        The key value to rank.

    Returns
    ----------
    The rank of `key` in `data`.
    """

    sort = np.argsort(data)
    rank = np.argwhere(sort == key)
    if rank.shape[0] > 1:
        rank = rank[0]

    return 256 - int(rank[0])


def guessDistance(data: np.ndarray,
                  key: int) -> float:
    """
    Computes the guessing distance of `data` at `key`, i.e.,
    the distance between the correcy key and the second guess.

    Parameters
    ----------
    `data` : array_like
        The data to compute the guessing distance for.
    `key`  : int
        The key value to compute the guessing distance at.

    Returns
    ----------
    The guessing distance of `data` at `key`.
    """

    return (data[key] - np.max(data[data != data[key]])) / (max(data) - min(data))


def guessEntropy(ranks: np.ndarray) -> float:
    """
    Computes the guessing entropy of `ranks`, i.e,
    the average rank of the correct key.

    Parameters
    ----------
    `ranks` : array_like
        The data ranks to compute the guessing entropy for.

    Returns
    ----------
    The guessing entropy of `ranks`.
    """

    return np.mean(ranks, axis=0)[-1]


def guessEntropyTo1(ranks: np.ndarray) -> int:
    """
    Computes the number of traces needed to reach guessing entropy 1.

    Parameters  
    ----------
    `ranks` : array_like
        The data ranks to compute the guessing entropy for.

    Returns
    ----------
    The number of traces needed to reach guessing entropy 1.
    """
    
    ge = np.mean(ranks, axis=0)
    return next((i for i, x in enumerate(ge) if x and np.all(ge[i:] == 1)), -1)


def successRate(ranks: np.ndarray) -> tuple[float, int]:
    """
    Computes the success rate of `ranks`, i.e.,
    the percentage of attacks for which the correct key is the first guess.

    Parameters
    ----------
    `ranks` : array_like
        The data ranks to compute the success rate for.

    Returns
    ----------
    A tuple like `(success_rate, n_success)`.
    """

    return np.mean(ranks[:, -1] == 1), np.sum(ranks[:, -1] == 1)


def guessMetrics(predictions: np.ndarray,
                 key: int) -> tuple[np.ndarray, float]:
    """
    Computes attack metrics, i.e, rank and guessing distance.
    Rank is a list over the number of traces.
    All other metrics can be computed from rank.

    Parameters
    ----------
    `predictions` : array_like
        A numpy array of shape (#traces, #key_values) with the log probabilities of the key guessings.
    `key` : int
        The value of the real key byte to guess.

    Returns
    ----------
    A tuple like `(rank, guessing-distance)`.
    """

    real_key_rank = []
    data = np.cumsum(predictions, axis=0)
    for i in np.arange(data.shape[0]):
        real_key_rank.append(rankKey(data[i, :], key))

    guessing_distance = guessDistance(data[-1, :], key)

    return np.array(real_key_rank), guessing_distance


def dumpMetrics(dir_path: str,
                gd: int,
                ranks: np.ndarray):
    """
    Save in `dir_path` a .txt with the principal results and a .npy with the ranks.\n
    Results are:
    - N. attacking traces
    - N. different keys
    - Guessing Entropy
    - Guessing Distance
    - N. traces to get GE=1
    - Success Rate

    Parameters
    ----------
    `dir_path` : str
        Directory where to save the results.
    `gd`    : int
        Guessing distance to save.
    `ranks` : array-like
        Ranks matrix to save. Used to compute the other metrics.
    """

    ge = guessEntropy(ranks)
    t1 = guessEntropyTo1(ranks)
    sr, ns = successRate(ranks)

    with open(dir_path + "attack_metrics.txt", "w") as file:
        file.write(f"N. attacking traces: {ranks.shape[1]}\n")
        file.write(f"N. different keys: {ranks.shape[0]}\n\n")
        file.write(f"Guessing Entropy: {ge}\n")
        file.write(f"Guessing Distance: {gd}\n")
        file.write(f"N. traces to GE=1: {t1}\n")
        file.write(f"Success Rate: {sr} ({ns}\\{ranks.shape[0]})\n")

    np.save(dir_path + "ranks.npy", ranks)
