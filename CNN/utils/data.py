"""
Authors : 
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari,
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

import CNN.modules as modules


def build_datamodule(data_config):
    datamodule_name = data_config['datamodule']['name']
    datamodule_config = data_config['datamodule']['config']
    dataset_dir = datamodule_config['dataset_dir']
    batch_size = datamodule_config['batch_size']
    num_workers = datamodule_config['num_workers']
    frequencies = datamodule_config['frequencies']

    datamodule_class = getattr(modules, datamodule_name)
    datamodule = datamodule_class(
        dataset_dir, batch_size, num_workers, frequencies)
    return datamodule
