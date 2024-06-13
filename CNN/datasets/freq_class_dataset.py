"""
Authors : 
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari,
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

import os
import numpy as np
from torch.utils.data import Dataset


class FrequencyClassifierDataset(Dataset):
    def __init__(self, data_dir, frequencies, which_subset='train'):

        self.frequencies = frequencies

        print("Loading {} set..".format(which_subset))

        self.windows = np.load(
            os.path.join(data_dir, '{}_windows.npy'.format(which_subset)))

        self.target = np.load(
            os.path.join(data_dir, '{}_targets.npy'.format(which_subset)))
        print("Number of windows in the {} set: {}".format(
            which_subset, self.target.shape[0]))

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, index):
        x = self.windows[index]
        which_freq = self.frequencies[int(self.target[index][0])]
        y = int(which_freq)
        return x, y
