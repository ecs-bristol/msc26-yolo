# Dataset

## Overview

The project uses a fire and smoke detection dataset obtained from Kaggle.

The dataset is used to train and evaluate four YOLO detector variants:

- YOLOv8-S
- YOLOv9-S
- YOLO11-S
- YOLO26-S

The detection task focuses on the fire and smoke classes.

## Dataset Structure

The dataset is organized into separate training, validation, and test subsets.

The YOLO dataset configuration is provided in:

`configs/data.yaml`

The complete image dataset is not stored in this repository because of its size. The configuration file is retained so that the dataset structure used by the training and validation scripts remains documented.

## Use in Model Training

The same dataset was used for the four-model comparison.

The main nominal training configuration was:

| Parameter | Value |
|---|---:|
| Image size | 640 |
| Epochs | 100 |
| Random seed | 42 |
| Nominal batch size | 16 |

YOLOv9-S was trained with a batch size of 8 in the completed experiment. The other three models used a batch size of 16.

This difference is retained in the repository so that the recorded experiment matches the model that was actually trained and evaluated.

## Validation and Test Data

The validation subset is used to calculate standard object-detection metrics including:

- Precision
- Recall
- mAP@50
- mAP@50-95

The test and benchmark material is used separately for runtime and scenario-based evaluation on PC and Jetson platforms.

## INT8 Calibration Data

TensorRT INT8 conversion requires representative calibration data.

A representative subset of the fire and smoke dataset was prepared for INT8 calibration. Its dataset configuration is provided in:

`configs/int8_calibration.yaml`

This calibration configuration is separate from the standard training configuration because its purpose is TensorRT INT8 model calibration rather than model training.

## Repository Scope

The repository includes:

- Dataset configuration files
- Training notebooks
- Evaluation scripts
- Optimization scripts
- Selected trained and optimized models
- Selected experimental results

The complete Kaggle image dataset and raw benchmark videos are not included.
