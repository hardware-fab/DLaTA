"""
Authors : 
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari,
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

import numpy as np
import pickle
from typing import Union

__hw = [bin(x).count("1") for x in range(256)]

hw = np.array(__hw)


def kahanSum(sum_: np.ndarray,
             c: np.ndarray,
             element: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the sum of `element` and `sum_`, using the Kahan summation algorithm.

    Parameters
    ----------
    `sum_`    : array_like
        The current running sum.
    `c`       : array_like
        The previous correction term.
    `element` : array_like
        The next element to add.

    Returns
    ----------
    The new running sum and the new correction term.

    Raises
    ------
    `AssertionError`
        If parameters shapes are not equal.
    """

    assert sum_.shape == c.shape, \
        f"sum_ and c shape must be equal: sum_ {sum_.shape}, c {c.shape}"
    assert element.shape == c.shape, \
        f"element and c shape must be equal: element {element.shape}, c {c.shape}"

    y = element - c
    t = sum_ + y
    c = (t - sum_) - y

    sum_ = t

    return sum_, c


def intToBytes(integer: int,
               num_bytes: int = None) -> np.ndarray:
    """
    Convert an integer into a numpy array of bytes.

    Parameters
    ----------
    `integer` : int
        The integer to be converted.
    `num_bytes` : int, optional
        The number of bytes to use for the conversion (default is the minimum number of bytes).

    Returns
    ----------
    A numpy array of unsigned 8-bit integers (dtype=np.uint8) that represents
    the byte arrays of the input integer.
    """

    if num_bytes is None:
        # Compute the number of bytes required
        num_bytes = (integer.bit_length() + 7) // 8
    bytes = list(integer.to_bytes(num_bytes, 'big'))
    return np.array(bytes, dtype=np.uint8)


def bytesToInt(bytes_: Union[bytes, list[int]]) -> int:
    """
    Convert a list of bytes into an integer.

    Parameters
    ----------
    `bytes_` : bytes | list[int8]
        A list of bytes to be converted.

    Returns
    ----------
    An integer that represents the concatenation of the input bytes.
    """

    integer = int.from_bytes(bytes_, byteorder='big')
    return integer
