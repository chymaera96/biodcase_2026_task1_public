# BioDCASE 2025 Task 1: Multi-Channel Alignment
Evaluation and Baseline System for Multi-Channel Alignment Task as part of BioDCASE 2025.

## Problem Summary

Researchers often deploy multiple audio recorders simultaneously, for example with passive automated recording units (ARU's) or embedded in animal-borne bio-loggers. Analysing sounds simultaneously captured by multiple recorders can provide insights into animal positions and numbers, as well as the dynamics of communication in groups. However, many of these devices are susceptible to desynchronization due to nonlinear clock drift, which can diminish researchers' ability to glean useful insights. Therefore, a reliable, post-processing-based re-synchronization method would increase usability of collected data.

In this challenge, participants will be presented with pairs of temporally desynchronized recordings and asked to design a system to synchronize them in time. In the development phase, participants will be provided audio pairs and a small set of ground-truth synchronization keypoints--the likes of which could be produced by a manual review of the data. In the evaluation phase, participants' systems will be ranked by their ability to synchronize unseen audio pairs.

## Task Description

Each dataset consists of a set of stereo audio files. The audio in the two channels of each audio file are not synchronized in time, due to non-linear clock drift. Each audio file has a corresponding set of annotations $k_0,\dots,k_{114}$ called _keypoints_. Each keypoint $k_i=(k_{i,0}, k_{i,1})$ consists of a timestamp $k_{i,0}$ for Channel 0 and a timestamp $k_{i,1}$ for Channel 1. The timestamps in each channel correspond to the same time in the physical world, but due to clock drift they do not appear at the same time in the recordings. The timestamps $k_{i,0}$ in Channel 0 occur at 1-second intervals. Timestamps always occur during the actual duration of the audio file, which means that for some files there are timestamps repeated at the beginning (to avoid negative timestamps) or at the end (to avoid exceeding the duration of the audio).

![A stereo audio waveform with overlaid keypoints.](https://github.com/earthspecies/biodcase_2025_task1/blob/main/keypoints.jpeg?raw=true)

During training, systems have access to keypoints' timestamps in both Channels 0 and 1. During inference, systems have access only to keypoints' timestamps in Channel 0, and must predict the corresponding Channel 1 timestamps. Systems are evaluated based on mean squared error (MSE) of their predicted Channel 1 timestamps, compared to ground-truth Channel 1 timestamps.

This repository's evaluator reports both mean squared error (MSE, in sec$^2$) and mean absolute error (MAE, in ms). In practice, we typically summarize performance using MAE (ms) because it's easier to interpret.

## Datasets

The challenge uses two datasets: `aru` and `zebra_finch`. The train and validation (val) portions of these datasets, which include audio and ground-truth keypoints, can be found [here](https://zenodo.org/records/15085675). The test portion, which includes only audio, will be provided during the evaluation phase of BioDCASE 2025. The domain shift between train and validation sets reflects the domain shift between train and evaluation sets.

In both datasets, desynchronization includes a constant shift in time between the two channels, as well as non-linear clock drift within each file. The total desynchronization never exceeds $\pm 5$ seconds.

The directory structure of the formatted datasets is:

```
formatted_data
├── aru
│   ├── train
│   │   ├── annotations.csv
│   │   └── audio
│   │       └── *.wav
│   └── val
│       ├── annotations.csv
│       └── audio
│           └── *.wav
└── zebra_finch
    ├── train
    │   ├── annotations.csv
    │   └── audio
    │       └── *.wav
    └── val
        ├── annotations.csv
        └── audio
            └── *.wav
```

## Requirements
This repository was tested with Python 3.11. Please see `requirements.txt` for package requirements.

The `deeplearning` baseline requires weights for the [BEATs](https://arxiv.org/abs/2212.09058) feature extractor, which can be obtained [here](https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea).

## Usage
Baselines included:
- `nosync`, in which no synchronization is performed
- `gccphat`, a signal-processing baseline based on GCC-PHAT

This repository also includes a deep learning system (`deeplearning`) for learning-based alignment.

For example usage, see `run_baselines.sh`.

If you want to reproduce results, do the following:
1. Download the dataset.
2. Download the BEATs checkpoint from the link above and place it in this folder.
3. Run ```bash run_baselines.sh /path/to/formatted_data```. (replace bash with your shell if necessary)
4. Results for each baseline method will be saved in a folder with a name like "BASELINEMETHOD_DATASET_val", for e.g. "deeplearning_baseline_zebra_finch_val". Predictions for each sample in a "predictions.csv" will be saved in each folder. The results of the evaluation metric will be saved as "predictions_evaluation.yaml".

For evaluation, model outputs are expected to be in the same format as the provided keypoint annotations, i.e. a `.csv` file with three columns `Filename`, `Time Channel 0`, and `Time Channel 1`. Outputs can be evaluated using `python evaluate.py --predictions-fp=/path/to/predictions.csv --ground-truth-fp=/path/to/ground/truth.csv`.

## Deep learning baseline description

### Overview

The deep learning system is a binary classifier trained to score whether a pair of mono audio windows (one from each channel) are aligned in time. At inference time, the model is used inside an offset search procedure to predict the Channel 1 timestamp for each Channel 0 keypoint.

### Technical details

For each window, features are extracted using a pre-trained [BEATs encoder](https://arxiv.org/abs/2212.09058) (optionally unfreezing the last transformer layers during training). The two embeddings are combined as $[z_0, z_1, |z_0-z_1|]$ and passed through a small MLP head trained with binary cross-entropy.

Inference uses a coarse-to-fine offset grid search per keypoint:
- For a subset of keypoints (controlled by `--keypoint-stride`), score candidate offsets on a coarse grid and select a smooth offset path using dynamic programming.
- Interpolate this coarse path to all keypoints.
- For each keypoint, refine by scoring a fine grid around the interpolated offset.

## Results of baseline systems on validation set

We report both MAE (ms) and MSE (sec$^2$) on validation sets; lower is better and perfect alignment is achieved when both metrics equal $0$.

| Model | aru MAE (ms) | aru MSE (sec$^2$) | zebra_finch MAE (ms) | zebra_finch MSE (sec$^2$) |
| --- | ---: | ---: | ---: | ---: |
| nosync | 116.2 | 0.019 | 2297.4 | 5.734 |
| gccphat | 122.4 | 0.020 | 2457.0 | 7.995 |
| deeplearning | 164.3 | 0.040 | 757.1 | 1.218 |


We conducted baseline experiments with `CUDA=13.0` on one H100 GPU. We verified that results are reproducible within this environment, but may not be when using different versions of `CUDA` or different GPU hardware.

## Code in this repository:

- `baseline_gccphat.py`: Inference using GCC-PHAT baseline.
- `baseline_deeplearning_inference.py`: Inference using deep learning baseline; assumes this has already been trained.
- `baseline_deeplearning_training.py`: Trains deep learning baseline.
- `baseline_nosync.py`: Inference using baseline that performs no synchronization.
- `beats.py`: Audio feature extractor code for deep learning baseline.
- `evaluate.py`: Evaluates model predictions, which are expected to be in the same format as the provided keypoint annotations.
- `models.py`: Model code for deep learning baseline.
- `run_baselines.sh`: Shell script to reproduce baseline results.
- `utils.py`: Helper functions for baseline systems.
