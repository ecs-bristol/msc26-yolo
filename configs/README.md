# Configuration Files

This directory contains configuration files used by the training, validation, and optimization workflows.

## Files

### `data.yaml`

Dataset configuration used for standard YOLO training and validation.

It defines the training, validation, and test dataset paths together with the fire and smoke class information.

### `int8_calibration.yaml`

Dataset configuration used for TensorRT INT8 calibration.

A representative subset of the fire and smoke dataset is used during INT8 calibration to support generation of the optimized TensorRT model.

## Dataset Documentation

Additional information about the dataset source, dataset structure, and experimental usage is provided in:

`docs/DATASET.md`
