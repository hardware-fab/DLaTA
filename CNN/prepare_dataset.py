"""
Authors : 
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari,
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

from tqdm import tqdm
import os
import numpy as np
from npy_append_array import NpyAppendArray
import h5py

np.random.seed(2024)


frequencies = {35: 0, 40: 1, 45: 2, 50: 3, 55: 4, 60: 5}


def _dataLoader(data_dir: str,
                batch_size: int):

    with h5py.File(data_dir, 'r') as f:
        traces = f['profiling/traces']
        target = f['profiling/labels']
        num_samples = traces.shape[0]
        num_batches = num_samples // batch_size
        for i in tqdm(range(num_batches), desc='Creating dataset'):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            yield traces[start_idx:end_idx], target[start_idx:end_idx]

        # Handle remaining samples
        if num_samples % batch_size != 0:
            start_idx = num_batches * batch_size
            end_idx = num_samples
            yield traces[start_idx:end_idx], target[start_idx:end_idx]


def createDataset(dataset_dir: str,
                  out_data_dir: str,
                  windows_per_trace: int = 2,
                  window_size: int = 500,
                  split_traces: float = 0.8,
                  min_slice: int = 1000):
    '''
    Create dataset from the original collection of traces.

    Parameters
    ----------
    `dataset_dir` : str
        Path to the DFS_DESYNCH dataset.
    `out_data_dir` : str
        Path to the output directory.
    `windows_per_trace` : int, optional
        Number of windows to extract from each key value (default is 2)
    `window_size` : int, optional
        Size of the window in sample (default is 500).
    `split_traces` : float, optional
        The proportion of traces to use for training (default is 0.8).
    `min_slice` : int, optional
        Minimum size of the slice (default is 1000).
    '''

    batch_size = 500  # Num of traces to for key value
    n_train_traces = int(batch_size * split_traces)
    n_valid_traces = (batch_size - n_train_traces) // 2

    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    train_win_path = os.path.join(out_data_dir, 'train_windows.npy')
    train_tar_path = os.path.join(out_data_dir, 'train_targets.npy')
    valid_win_path = os.path.join(out_data_dir, 'valid_windows.npy')
    valid_tar_path = os.path.join(out_data_dir, 'valid_targets.npy')
    test_win_path = os.path.join(out_data_dir, 'test_windows.npy')
    test_tar_path = os.path.join(out_data_dir, 'test_targets.npy')

    with NpyAppendArray(train_win_path, delete_if_exists=True) as npa_train_win, \
            NpyAppendArray(train_tar_path, delete_if_exists=True) as npa_train_tar, \
            NpyAppendArray(valid_win_path, delete_if_exists=True) as npa_valid_win, \
            NpyAppendArray(valid_tar_path, delete_if_exists=True) as npa_valid_tar, \
            NpyAppendArray(test_win_path, delete_if_exists=True) as npa_test_win,   \
            NpyAppendArray(test_tar_path, delete_if_exists=True) as npa_test_tar:

        for all_traces, all_targets in _dataLoader(dataset_dir, batch_size):
            
            # Training
            traces = all_traces[:n_train_traces]
            targets = all_targets[:n_train_traces]
            for t_idx, trace in tqdm(enumerate(traces), leave=False, desc='Training'):
                target = targets[t_idx]
                old_subsets = []
                for _ in range(windows_per_trace):
                    found = False
                    while not found:
                        which_subset = np.random.choice(range(len(target[0])))
                        if which_subset in old_subsets:
                            # This allows a maximum of num frequencies per trace
                            continue
                        if which_subset == len(target[0])-1:
                            end = traces.shape[1]
                        else:
                            end = target[0][which_subset+1]
                        win_len = end - target[0][which_subset]
                        if win_len < min_slice:
                            continue
                        start_idx = (win_len // 2) - (window_size // 2)
                        start_idx = start_idx + target[0][which_subset]
                        window = trace[start_idx:start_idx+window_size]
                        if len(window) == window_size:
                            found = True
                            old_subsets.append(which_subset)
                            npa_train_win.append(np.expand_dims(window, 0))
                            npa_train_tar.append(np.reshape(
                                target[1][which_subset], [-1, 1]))

            # Validation
            traces = all_traces[n_train_traces:n_train_traces+n_valid_traces]
            targets = all_targets[n_train_traces:n_train_traces+n_valid_traces]
            for t_idx, trace in tqdm(enumerate(traces), leave=False, desc='Validation'):
                target = targets[t_idx]
                old_subsets = []
                for _ in range(windows_per_trace):
                    found = False
                    while not found:
                        which_subset = np.random.choice(range(len(target[0])))
                        if which_subset in old_subsets:
                            continue
                        if which_subset == len(target[0])-1:
                            end = traces.shape[1]
                        else:
                            end = target[0][which_subset+1]
                        win_len = end - target[0][which_subset]
                        if win_len < min_slice:
                            continue
                        start_idx = (win_len // 2) - (window_size // 2)
                        start_idx = start_idx + target[0][which_subset]
                        window = trace[start_idx:start_idx+window_size]
                        if len(window) == window_size:
                            found = True
                            old_subsets.append(which_subset)
                            npa_valid_win.append(np.expand_dims(window, 0))
                            npa_valid_tar.append(np.reshape(
                                target[1][which_subset], [-1, 1]))

            # Test
            traces = all_traces[n_train_traces+n_valid_traces:]
            targets = all_targets[n_train_traces+n_valid_traces:]
            for t_idx, trace in tqdm(enumerate(traces), leave=False, desc='Testing'):
                target = targets[t_idx]
                found = False
                old_subsets = []
                for _ in range(windows_per_trace):
                    found = False
                    while not found:
                        which_subset = np.random.choice(range(len(target[0])))
                        if which_subset in old_subsets:
                            continue
                        if which_subset == len(target[0])-1:
                            end = traces.shape[1]
                        else:
                            end = target[0][which_subset+1]
                        win_len = end - target[0][which_subset]
                        if win_len < min_slice:
                            continue
                        start_idx = (win_len // 2) - (window_size // 2)
                        start_idx = start_idx + target[0][which_subset]
                        window = trace[start_idx:start_idx+window_size]
                        if len(window) == window_size:
                            found = True
                            old_subsets.append(which_subset)
                            npa_test_win.append(np.expand_dims(window, 0))
                            npa_test_tar.append(np.reshape(
                                target[1][which_subset], [-1, 1]))
