"""
Authors : 
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari,
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

import numpy as np
from scipy.interpolate import interp1d


class FrequencyRescaler:
    '''
    Rescales a window into another frequency.
    '''

    def __init__(self, freq_ratio, interp_kind='linear'):
        self.freq_ratio = freq_ratio
        self.interp_kind = interp_kind

    def scale_windows(self, windows):
        '''
        Rescale a window/trace to another frequency using interpolation.
        '''
        windows_size = windows.shape[-1]
        time = np.arange(0, windows_size)
        f_interp = interp1d(time, windows, kind=self.interp_kind)
        step = self.freq_ratio
        n_steps = int(windows_size / step) + 1
        time_interp = np.linspace(0, windows_size-1, n_steps)
        windows_interp = f_interp(time_interp)
        return windows_interp.astype(np.float32)
