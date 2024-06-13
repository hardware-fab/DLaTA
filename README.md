# A Deep Learning-assisted Template Attack Against Dynamic Frequency Scaling Countermeasures

To run an experiment, follow the Jupyter Notebook `DLaTA.ipynb`.

## Organization

The repository is organized as follows:

- `/CNN`: Contains modules and configuration files for the Convolutional Neural Network use for classify the operating frequency.
- `/templateAttack`: Contains functions and classes for profiling a Template Attack and attacking side-channel traces.
- `/utils`: Contains functions for segmenting and interpolating side-channel traces.
- `DLaTA`: Jupyter Notebook for run a demo.

```txt
.
├── CNN
│   ├── configs/
│   ├── datasets
│   │   └── freq_class_dataset.py
│   ├── models
│   │   ├── custom_layers.py
│   │   ├── resnet.py
│   │   └── resnet_time_series_classifier.py
│   ├── modules
│   │   ├── freq_class_datamodule.py
│   │   └── freq_class_module.py
│   ├── prepare_dataset.py
│   ├── train.py
│   └── utils
│       ├── data.py
│       ├── logging.py
│       ├── module.py
│       ├── trainer.py
│       └── utils.py
├── templateAttack
│   ├── aesSca.py
│   ├── metrics.py
│   ├── preprocess.py
│   ├── TemplateAttack.py
│   └── utils.py
├── utils
│    ├── rescaler.py
│    └── segmentation.py
└── DLaTA.ipynb
```

## Dataset

The `DFS_DESYNCH` dataset is avaible [here](https://dx.doi.org/10.21227/8txt-xe26).

The dataset is stored in the HDF5 file format (`dfs_desynch.h5`) and has the following structure:

- **Profiling and Attack groups:** The traces are separated into two main groups: "profiling" and "attack". Each group contains 128,000 traces for a total of 256,000 traces.
- **Three Datasets per group:** Each group ("profiling" and "attack") consists of three internal datasets:
  - **traces:** This dataset includes 128,000 power traces, each containing 200,000 time samples. The traces capture the entire AES encryption process preceded by a sequence of random instructions. The traces are pre-processed with a high-pass filter with a 125 kHz cut-off frequency.
  - **labels:** This dataset provides labels for each power trace in the "traces" dataset, indicating the frequency changes that occurred during the measurement. Each label has two fields:
    - `sample`: This field denotes the time sample at which a frequency change happens, with values ranging from 0 to 200,000.
    - `frequency`: This field specifies the new operating frequency starting from the corresponding sample. It can take values from the set {35, 40, 45, 50, 55, 60}.
  - **metadata:** This dataset contains metadata for each trace, including two members:
    - `key`: The secret key used for AES encryption.
    - `plaintext`: The plaintext used for the AES encryption.

## Note

This work is part of [1].

This repository is protected by copyright and licensed under the [Apache-2.0 license](https://github.com/hardware-fab/DLaTA/blob/main/LICENSE) file.

© 2024 hardware-fab

> [1] Galli, D., Lattari, F., Matteucci, M., & Zoni, D.. A Deep Learning-assisted Template Attack Against Dynamic Frequency Scaling Countermeasures.
