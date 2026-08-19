# Reproducibility

## Overview

This document describes the recommended order for reproducing the main experimental workflow in this repository.

The workflow is divided into model training, model evaluation, Jetson deployment, optimization, validation, and final application testing.

Exact JetPack, CUDA, TensorRT, PyTorch, and other platform-specific package versions should match the target Jetson environment. TensorRT engine files may need to be regenerated when the software or hardware environment changes.

## 1. Prepare the Dataset

Obtain the fire and smoke dataset used for the project and arrange the training, validation, and test subsets according to the dataset configuration.

Standard dataset configuration:

`configs/data.yaml`

INT8 calibration configuration:

`configs/int8_calibration.yaml`

Additional dataset information is provided in:

`docs/DATASET.md`

## 2. Install the PC Training Environment

Install the main Python dependencies:

```bash
pip install -r requirements.txt
```

The four training notebooks are located under:

`training/`

## 3. Train the Four YOLO Models

Run the training notebooks for:

- YOLOv8-S
- YOLOv9-S
- YOLO11-S
- YOLO26-S

The completed experiment used:

- image size: 640
- epochs: 100
- seed: 42
- batch size: 16 for YOLOv8-S, YOLO11-S, and YOLO26-S
- batch size: 8 for YOLOv9-S

Selected trained model files are stored under:

`models/`

## 4. Run the Four-Model Comparison

Use the comparison workflow under:

`evaluation/model_comparison/`

Compare the trained models using metrics such as:

- Precision
- Recall
- mAP@50
- mAP@50-95
- Model size

YOLO11-S is the detector selected for the later optimization stage in this project.

## 5. Run PC Runtime Benchmarks

PC benchmark scripts are located under:

`evaluation/pc/`

Run the four trained models using consistent benchmark scenarios and inference settings.

Collect the required runtime summary outputs for later comparison.

## 6. Prepare the Jetson Platform

Prepare the NVIDIA Jetson Orin Nano before running edge benchmarks.

The deployment procedure is documented in:

`deployment/README.md`

The procedure includes:

1. Preparing the microSD system image
2. Booting and configuring the Jetson system
3. Verifying the Python and inference environment
4. Verifying CUDA and TensorRT support
5. Installing or verifying PyTorch, Ultralytics, OpenCV, and required dependencies
6. Connecting and testing the USB camera
7. Transferring the project files and models to the Jetson device

Jetson-specific Python dependencies are summarized in:

`requirements_jetson.txt`

## 7. Run Jetson Runtime Benchmarks

Jetson benchmark scripts are located under:

`evaluation/jetson/`

Use the same main scenario categories used for PC testing:

- Fire
- Smoke
- Mixed fire and smoke
- Negative scene

Collect FPS, latency, and detection summary results.

## 8. Run YOLO11-S Edge Optimization

YOLO11-S is used for the optimization stage.

### FP16

Use:

`optimization/tensorrt/export_fp16_tensorrt.py`

### INT8

Use:

`optimization/tensorrt/export_int8_tensorrt.py`

Representative calibration data is defined using:

`configs/int8_calibration.yaml`

### P20 Pruning

Use the pruning workflow under:

`optimization/pruning/`

Selected pruned and TensorRT model files are stored under:

`models/`

## 9. Validate Optimized Models

Use the scripts under:

`optimization/evaluation/`

The main evaluation scripts include:

- `validate_model.py`
- `threshold_sweep.py`
- `compare_summaries.py`
- `make_optimization_table.py`

Evaluate both accuracy-related and runtime-related effects of optimization.

## 10. Run the Final Application

The final fire and smoke monitoring application is stored under:

`application/`

The application combines the YOLO detector with the MobileNetV3-Small scene classifier and real-time alarm logic.

The scene classifier can be enabled or disabled to support comparison of the two operating modes.

## 11. Review Results

Selected experimental summaries and figures are stored under:

`results/`

The results should be interpreted together with the experiment design documented in:

`docs/EXPERIMENT_PROTOCOL.md`

## Repository Reproduction Order

```text
Dataset preparation
        ↓
Four-model training
        ↓
Four-model accuracy comparison
        ↓
PC runtime benchmark
        ↓
Jetson environment deployment
        ↓
Jetson runtime benchmark
        ↓
YOLO11-S optimization
        ↓
Optimized-model validation
        ↓
Final monitoring application
        ↓
Result comparison
```
