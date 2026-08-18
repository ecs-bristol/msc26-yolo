# Configuration Files

This directory stores lightweight dataset and deployment configuration files.

## Expected Files

### `data.yaml`

Original YOLO dataset configuration used for training and validation.

The file should define the dataset paths and class names required by Ultralytics YOLO. Class order should follow the original dataset definition and should not be changed manually unless the dataset itself is changed.

### `int8_calibration.yaml`

Dataset configuration used during TensorRT INT8 calibration/export.

This configuration should point to the representative calibration image set prepared for INT8 quantization. Calibration data should reflect the expected fire, smoke, mixed, and negative deployment scenes as closely as possible.

## Dataset Storage

The complete dataset is intentionally not stored in this repository. Only YAML configuration files are version controlled.
