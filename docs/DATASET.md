# Dataset

## Source

The fire and smoke detection dataset used in this project was obtained from Kaggle.

Dataset identifier:

```text
sayedgamal99/smoke-fire-detection-yolo
```

## Dataset Organization

The downloaded dataset provides data used for training, validation, and testing. The exact directory structure is defined by the dataset YAML file.

The repository keeps only the relevant YAML configuration files under `configs/`; the full image and label dataset is not committed to GitHub.

## YOLO Labels

The training notebooks use the dataset's YOLO-format annotations. Dataset class names and class order should always be read from the original YAML configuration rather than redefined from memory.

## Repository Files

```text
configs/data.yaml
```

Original dataset configuration used for training and validation.

```text
configs/int8_calibration.yaml
```

Configuration used for representative INT8 calibration data on the Jetson optimization workflow.

## Data Integrity Checks

Before formal training, the workflow checks:

- Dataset directory structure
- YAML paths
- Image/label pairing
- YOLO-format bounding-box coordinates
- Class identifiers
- Example annotated images

Negative images without fire or smoke may have empty/no object annotations depending on the dataset design.

## Dataset Licensing

This repository does not redistribute the full Kaggle dataset. Users should consult the original Kaggle dataset page for the applicable dataset license and usage conditions.
