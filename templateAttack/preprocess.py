"""
Authors : 
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari,
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

import numpy as np
import scipy.signal


def highpass(traces: np.ndarray,
             Wn: float = 0.002) -> np.ndarray:
    """
    Applies a 3rd-order high-pass filter to the traces.

    Parameters
    ----------
    `traces`: array_like
        The traces to filter.
    `Wn`    : float, optional
        The cutoff frequency of the filter, express relative to the Nyquist frequency (default is 0.002).

    Returns
    ----------
    The filtered traces.
    """

    b, a = scipy.signal.butter(3, Wn)
    y = scipy.signal.filtfilt(b, a, traces).astype(np.float32)

    return (traces - y).astype(np.float32)


def aggregate(trace: np.ndarray,
              n: int) -> np.ndarray:
    """
    Aggregates `n` consective sample of the trace.

    Parameters
    ----------
    `trace` : array_like
        The trace to aggregate.
    `n`     : int
        The number of sample to aggregate.

    Returns
    ----------
    The aggregated trace.
    """

    # Divide array into n subsections
    num_chunks = int(trace.shape[1] / n)
    chunks = np.array_split(trace, num_chunks, axis=1)
    # Compute the mean of each subsection
    means = [np.mean(chunk, axis=1, dtype=np.float32) for chunk in chunks]

    return np.array(means, dtype=np.float32).T
