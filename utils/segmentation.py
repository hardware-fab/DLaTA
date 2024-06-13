"""
Authors : 
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari,
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

import os

from tqdm.auto import tqdm

import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from npy_append_array import NpyAppendArray
from math import ceil

from scipy.ndimage import label
from itertools import repeat

from CNN.utils import (parse_arguments, get_neptune_run,
                       get_experiment_config_dir)
from utils import FrequencyRescaler

import CNN.modules as modules


FILE_EXISTS_MSG = "Files already exist:"
ABORTING_MSG = "aborting operation. If you want to override them, set 'force'."
OVERRIDE_MSG = "overriding files..."

FREQUENCIES = {35: 0, 40: 1, 45: 2, 50: 3, 55: 4, 60: 5}


def _segmentationGradCAM(traces, module, batch_size, num_frequencies, device):
    segmented_traces = []

    n_iters = ceil(traces.shape[0] / batch_size)

    torch.cuda.empty_cache()

    for i in tqdm(range(n_iters), desc='Grad-CAM Segmentation', leave=False, position=1):
        low_plain_idx = i*batch_size
        high_plain_idx = min((i+1)*batch_size, traces.shape[0])
        real_batch_size = high_plain_idx - low_plain_idx

        curr_traces = traces[low_plain_idx: high_plain_idx]
        curr_traces = curr_traces.reshape(real_batch_size, traces.shape[-1])
        curr_traces = torch.from_numpy(curr_traces)
        curr_traces = curr_traces.to(device)

        y_hat = module(curr_traces)

        maps = module.model.encoder.forward_hooks['grad_cam'].detach()
        num_maps = maps.shape[1]

        batch_weights = torch.zeros(
            (real_batch_size, num_frequencies, num_maps)).cuda()  # 8  x 6 x 32

        for k in range(num_frequencies):
            pred = y_hat[:, k].sum()
            pred.backward(retain_graph=True)
            # 8 x 32
            grad = module.model.encoder.backward_hooks['grad_cam'].detach()
            batch_weights[:, k, :] = grad  # 8 x 6 x 32
        batch_maps = maps.permute(0, 2, 1)  # maps: 8 x 134016 x 32
        class_maps = torch.bmm(
            batch_maps, batch_weights.permute(0, 2, 1))  # 8 x 134016 x 6
        class_maps = torch.nn.functional.relu(class_maps)

        del maps
        del batch_weights
        segmented_traces.append(class_maps.detach().cpu())
        del class_maps

    segmented_traces = torch.cat(segmented_traces, dim=0)
    segmented_traces = segmented_traces.permute(0, 2, 1)

    data_return = segmented_traces.data.numpy()

    return data_return


def _segmentationGT(traces, gts):

    trace_len = traces.shape[1]
    segmented_traces = np.zeros([len(gts), 6, trace_len])

    for gt_idx, gt in enumerate(gts):
        for win_idx, (sample, freq) in enumerate(zip(gt['sample'], gt['frequency'])):
            label = FREQUENCIES[int(freq)]

            start_idx = sample
            if win_idx == len(gt['sample'])-1:
                end_idx = trace_len
            else:
                end_idx = gt['sample'][win_idx+1]

            segmented_traces[gt_idx, label, start_idx:end_idx] = 1

    return segmented_traces


def _compute_win_batch(batch_size, num_plain_texts, plain_segm_batch, frequencies, num_classes, batch_idx):

    orig_batch_size = batch_size

    if (batch_idx * batch_size + batch_size) > num_plain_texts:
        batch_size = num_plain_texts - orig_batch_size

    df_list = []

    for plain_index in range(batch_idx * orig_batch_size, batch_idx * orig_batch_size + batch_size):
        if plain_index > num_plain_texts:
            continue

        plain_segm = plain_segm_batch[plain_index -
                                      (batch_idx * orig_batch_size)]

        for freq_idx in range(num_classes):

            curr_df = _compute_win(plain_segm, plain_index,
                                   frequencies, freq_idx)
            df_list.append(curr_df)

    return pd.concat(df_list, ignore_index=True)


def _compute_win(plain_segm, plain_idx, frequencies, freq_idx):
    df_windows = pd.DataFrame(
        columns=['plain_index', 'time_start', 'time_end', 'frequency'])
    class_segm = plain_segm == freq_idx
    if not np.count_nonzero(class_segm) == 0:
        class_segm = class_segm.astype(int)
        win_lclz, num_win = label(class_segm)

        for win_id in range(1, num_win+1):
            win_idx = np.where(win_lclz == win_id)[0]
            time_start = win_idx[0]
            time_end = win_idx[-1]
            df_windows = df_windows._append(
                {'plain_index': plain_idx,
                 'time_start': time_start,
                 'time_end': time_end+1,
                 'frequency': sorted(list(frequencies.keys()))[freq_idx]}, ignore_index=True)
    return df_windows


def _align_trace_batch(batch_size, num_plain_texts, df_windows, traces, batch_idx, target_freq, interp_kind):

    orig_batch_size = batch_size

    TRACE_SIZE = traces.shape[-1]

    if (batch_idx * batch_size + batch_size) > num_plain_texts:
        batch_size = num_plain_texts - orig_batch_size * batch_idx

    aligned_traces = np.zeros([batch_size, TRACE_SIZE])

    for plain_index in range(batch_idx * orig_batch_size, batch_idx * orig_batch_size + batch_size):
        if plain_index > num_plain_texts:
            continue

        aligned_traces[plain_index-(batch_idx * orig_batch_size), :] = _align_trace(
            traces[plain_index], df_windows, plain_index, target_freq, interp_kind)

    return batch_idx, aligned_traces


def _align_trace(trace, df_windows, plain_index, target_freq, interp_kind):
    df_plain = df_windows[df_windows['plain_index'] == plain_index].copy()
    df_plain.sort_values(by='time_start', inplace=True)

    time_indices = df_plain[['time_start', 'time_end']].to_numpy()
    frequencies = df_plain['frequency'].to_numpy()

    TRACE_SIZE = trace.shape[-1]

    n_switches = frequencies.shape[0]
    aligned_trace = np.zeros(TRACE_SIZE)

    start_idx = 0

    to_break = False

    for i in range(n_switches):

        curr_idx = time_indices[i]
        curr_start = curr_idx[0]
        curr_end = curr_idx[1]
        curr_size = curr_end - curr_start
        curr_freq = frequencies[i]

        if int(curr_freq) == int(target_freq):
            end_idx = start_idx + curr_size
            if start_idx + curr_size > TRACE_SIZE:
                end_idx = TRACE_SIZE
                curr_size = TRACE_SIZE - start_idx
                curr_end = curr_start + curr_size
                to_break = True
            aligned_trace[start_idx:end_idx] = trace[curr_start:curr_end]
            start_idx += curr_size
        elif curr_end - curr_start < 2:
            end_idx = start_idx + curr_size
            if start_idx + curr_size > TRACE_SIZE:
                end_idx = TRACE_SIZE
                curr_size = TRACE_SIZE - start_idx
                curr_end = curr_start + curr_size
                to_break = True
            aligned_trace[start_idx:end_idx] = trace[curr_start:curr_end]
            start_idx += curr_size
        else:
            freq_ratio = float(target_freq) / float(curr_freq)
            rescaler = FrequencyRescaler(freq_ratio, interp_kind)
            aligned_win = rescaler.scale_windows(trace[curr_start:curr_end])
            size_aligned = aligned_win.shape[-1]
            if start_idx + size_aligned > TRACE_SIZE:
                size_aligned = TRACE_SIZE - start_idx
                aligned_win = aligned_win[:size_aligned]
                to_break = True
            end_idx = start_idx + size_aligned
            aligned_trace[start_idx:end_idx] = aligned_win
            start_idx += size_aligned

        if to_break:
            break

    trace = aligned_trace[:trace.shape[-1]]
    return trace.astype(np.float32)


def _localize_and_align(segmented_traces, traces, frequencies,
                        df_windows, target_freq, interp_kind,
                        num_workers, process_pool):
    '''
    Work method of one process computing all plains windows localization
    '''
    mgr = mp.Manager()
    ns = mgr.Namespace()
    ns.df_list = []

    batch_size = traces.shape[0] // num_workers

    batch_idxs = [bi for bi in range(num_workers)]

    plain_segm_batch = []
    plain_segm_tot = np.argmax(segmented_traces, axis=1)
    for batch_idx in batch_idxs:
        plain_segm_batch.append(
            plain_segm_tot[batch_idx*batch_size:batch_idx*batch_size+batch_size])

    dfs = process_pool.starmap(
        _compute_win_batch,
        zip(repeat(batch_size), repeat(traces.shape[0]), plain_segm_batch,
            repeat(frequencies), repeat(len(frequencies)), batch_idxs))

    df_windows = pd.concat(dfs, ignore_index=True)

    out = process_pool.starmap(
        _align_trace_batch,
        zip(repeat(batch_size), repeat(traces.shape[0]), repeat(df_windows),
            repeat(traces), batch_idxs, repeat(target_freq), repeat(interp_kind)))

    aligned_traces_sorted = [x for _, x in sorted(out)]

    aligned_traces = np.concatenate(aligned_traces_sorted)

    return df_windows, aligned_traces


def getModule(SID: str,
              neptune_config: str = 'CNN/configs/common/neptune_configs.yaml') -> torch.nn.Module:
    """
    Get the best model from a Neptune Run and return the corresponding module.

    Parameters
    ----------
    `SID` : str
        The Neptune Run ID.
    `neptune_config` : str, optional
        The path to the Neptune configuration file (default is 'CNN/configs/common/neptune_configs.yaml').

    Returns
    -------
    The best model from the Neptune Run.
    """

    # Get Neptune Run (by SID)
    df = get_neptune_run(neptune_config, SID)

    # Get experiment name
    exp_name = df['sys/name'].iloc[0]

    # Get best model path
    best_model_ckpt = df['experiment/model/best_model_path'].iloc[0]

    # Get config dir
    config_dir = get_experiment_config_dir(best_model_ckpt, exp_name)

    _, module_config, __ = parse_arguments(config_dir)

    # Build Model
    # -----------
    module_name = module_config['module']['name']
    module_config = module_config['module']['config']
    # Activate Grad-CAM
    module_config['model']['config'][
        'encoder_params']['gradcam'] = True
    module_class = getattr(modules, module_name)
    module = module_class.load_from_checkpoint(best_model_ckpt,
                                               module_config=module_config)

    return module


def segmentAlignGradCAM(traces: np.ndarray,
                        module: torch.nn.Module,
                        output_dir: str,
                        *,  # Force keyword arguments
                        log: bool = False,
                        batch_size: int = 500,
                        gpu: int = 0,
                        num_workers: int = 10,
                        force: bool = False):
    """
    Segment the traces in the given dataset.

    Parameters
    ----------
    `traces` : np.ndarray
        The traces to segment.
    `module` : torch.nn.Module
        The model to use for the Gradcam segmentation.
    `output_dir` : str
        The path to the output directory.
    `log` : bool, optional
        Whether to log the segmentation and localization (default is False).
    `batch_size` : int, optional
        The batch size to use for the Gradcam segmentation (default is 500).
    `gpu` : int, optional
        The GPU to use for the Gradcam segmentation (default is 0).
    `num_workers` : int, optional
        The number of workers to use for the aligment (default is 10).
    `force` : bool, optional
        True if you want to override existing files (default is False).
    """

    output_dir = os.path.join(output_dir, 'grad-cam')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if _checkOverride(output_dir, force):
        return

    # Get Device
    device = torch.device(
        f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    # Set Module
    module.to(device)
    module.eval()
    # -----------

    # Logging
    segm_file = os.path.join(output_dir, 'segmentation.npy')

    # Segment and Align
    # -----------------
    process_pool = mp.Pool(processes=num_workers)
    with NpyAppendArray(segm_file, delete_if_exists=True) as npa_segment:
        with tqdm(desc='Aligning',total=len(traces)//batch_size, position=0) as pbar:
            for traces_batch in _dataLoader(traces, batch_size):
                segm = _segmentationGradCAM(traces_batch, module, batch_size//25,
                                            len(FREQUENCIES), device)
                if log:
                    npa_segment.append(segm)
                pbar.update(0.7)

                align(traces_batch, segm, output_dir, process_pool,
                    log=log, num_workers=num_workers)
                pbar.update(0.3)
                
    process_pool.close()


def segmentAlignGT(traces: np.ndarray,
                   frequencies_gt: np.ndarray,
                   output_dir: str,
                   *,  # Force keyword arguments
                   log: bool = False,
                   batch_size: int = 500,
                   num_workers: int = 10,
                   force: bool = False):
    """
    Segment and align the traces in the given dataset usging the ground-thruth.

    Parameters
    ----------
    `traces` : np.ndarray
        The traces to segment.
    `frequencies_gt` : np.ndarray
        The frequency labels to segment the traces to.
    `output_dir` : str
        The path to the output directory.
    `log` : bool, optional
        Whether to log the localization (default is False).
    `batch_size` : int, optional
        The batch size to use for the Gradcam segmentation (default is 500).
    `num_workers` : int, optional
        The number of workers to use for the aligment (default is 10).
    `force` : bool, optional
        True if you want to override existing files (default is False).
    """

    output_dir = os.path.join(output_dir, 'gt')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if _checkOverride(output_dir, force):
        return

    # Segment and Align
    # -----------------
    process_pool = mp.Pool(processes=num_workers)
    with tqdm(desc='Aligning', total=len(traces)//batch_size, position=0) as pbar:
        for traces_batch, frequencies_batch in _dataLoader(traces, batch_size, frequencies_gt):
            segm = _segmentationGT(traces_batch, frequencies_batch)
            pbar.update(0.2)
            align(traces_batch, segm, output_dir, process_pool,
                  log=log, num_workers=num_workers)
            pbar.update(0.8)

    process_pool.close()


def align(traces: np.ndarray,
          segments: np.ndarray,
          output_dir: str,
          process_pool: mp.Pool,
          *,  # Force keyword arguments
          log: bool = False,
          target_freq: int = 45,
          interp_kind: str = 'linear',
          num_workers: int = 10):
    '''
    Align the traces using the given segmentation.

    Parameters
    ----------
    `traces` : np.ndarray
        The traces to align.
    `segments` : np.ndarray
        The segments to use for the alignment.
    `output_dir` : str
        The path to the output directory.
    `process_pool` : mp.Pool
        The process pool to use for the alignment.
    `log` : bool, optional
        Whether to log the localization (default is False).
    `target_freq` : int, optional
        The target frequency to align the traces to (default is 45).
    `interp_kind` : str, optional
        The interpolation kind to use for the alignment (default is 'linear').
    `num_workers` : int, optional
        The number of workers to use for the alignment (default is 10).
    '''

    df_windows = pd.DataFrame(
        columns=['plain_index', 'time_start', 'time_end', 'frequency'])

    aligned_file = os.path.join(output_dir, 'traces_aligned.npy')
    lclz_file = os.path.join(output_dir, f'localization.csv')

    # Localization
    # ------------
    with NpyAppendArray(aligned_file, delete_if_exists=False) as npa_align:
        df_lclz, aligned_traces = _localize_and_align(
            segments, traces, FREQUENCIES, df_windows, target_freq,
            interp_kind, num_workers, process_pool)

        if log:
            df_lclz.to_csv(lclz_file, index=False, mode='a')

        npa_align.append(aligned_traces.astype(np.float32))


def _dataLoader(traces: np.ndarray,
                batch_size: int,
                frequency_labels: np.ndarray = None):

    for i in range(len(traces)//batch_size):
        batch_traces = traces[i*batch_size: (i+1)*batch_size]
        if frequency_labels is not None:
            batch_frequencies = frequency_labels[i *
                                                 batch_size: (i+1)*batch_size]

        if i == len(traces)//batch_size - 1:
            remaining_traces = len(traces) % batch_size
            if remaining_traces > 0:
                batch_traces = traces[i*batch_size:]
                if frequency_labels is not None:
                    batch_frequencies = frequency_labels[i*batch_size:]
        if frequency_labels is not None:
            yield batch_traces, batch_frequencies
        else:
            yield batch_traces


def _checkOverride(path, force):
    exit = False
    if os.listdir(path):
        print(FILE_EXISTS_MSG, end=" ")
        print(ABORTING_MSG) if not force else print(OVERRIDE_MSG)
        exit = False if force else True

    if force:
        _deleteFiles(path)

    return exit


def _deleteFiles(path):
    for file in os.listdir(path):
        os.remove(os.path.join(path, file))
