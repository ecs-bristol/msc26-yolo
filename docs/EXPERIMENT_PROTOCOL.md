# Experiment Protocol

## 1. Objective

The project investigates real-time fire and smoke detection using YOLO-based object detectors and edge deployment on an NVIDIA Jetson platform.

The experimental workflow consists of:

1. Training and comparing four YOLO detector variants
2. Evaluating runtime performance on PC and Jetson
3. Selecting YOLO11-S for edge optimization
4. Applying TensorRT FP16, TensorRT INT8, and P20 pruning
5. Re-evaluating optimized models
6. Integrating YOLO with a MobileNetV3-Small scene classifier
7. Running the final real-time fire and smoke monitoring application on Jetson

## 2. Compared Detection Models

The following models are included in the initial comparison:

- YOLOv8-S
- YOLOv9-S
- YOLO11-S
- YOLO26-S

### Training Configuration

| Model | Image Size | Epochs | Batch Size | Seed |
|---|---:|---:|---:|---:|
| YOLOv8-S | 640 | 100 | 16 | 42 |
| YOLOv9-S | 640 | 100 | 8 | 42 |
| YOLO11-S | 640 | 100 | 16 | 42 |
| YOLO26-S | 640 | 100 | 16 | 42 |

The same image size, number of epochs, and random seed were used across the four completed experiments. YOLOv9-S used batch size 8 in the completed training run.

Training notebooks are stored under:

`training/`

## 3. Accuracy Evaluation

The trained models are evaluated using standard object-detection metrics:

- Precision
- Recall
- mAP@50
- mAP@50-95
- Model size

The four-model comparison workflow is stored under:

`evaluation/model_comparison/`

The comparison is used to identify a model that provides a suitable balance between detection performance and deployment efficiency.

YOLO11-S was selected for the subsequent edge-optimization stage.

## 4. Runtime Evaluation

Runtime testing is carried out separately on a PC platform and an NVIDIA Jetson Orin Nano.

### PC Evaluation

PC benchmark scripts are stored under:

`evaluation/pc/`

The local PC platform used for the project includes an NVIDIA GTX 1660 Ti GPU.

### Jetson Evaluation

Jetson benchmark scripts are stored under:

`evaluation/jetson/`

The target edge platform is an NVIDIA Jetson Orin Nano.

### Runtime Metrics

The benchmark workflow records metrics such as:

- End-to-end FPS
- Mean inference latency
- Pre-processing latency
- Post-processing latency
- Detection counts
- Fire detection counts
- Smoke detection counts
- Detection confidence
- Model size

## 5. Scenario-Based Testing

Runtime testing uses representative fire and smoke scenarios.

The main scenario categories are:

- Fire
- Smoke
- Mixed fire and smoke
- Negative scene

The same scenario categories are used to support consistent model and platform comparison.

## 6. YOLO11-S Optimization

After the initial model comparison, YOLO11-S is used as the main detector for edge optimization.

The optimization workflow includes:

### TensorRT FP16

FP16 conversion is used to improve inference efficiency on Jetson.

Scripts are stored under:

`optimization/tensorrt/`

### TensorRT INT8

INT8 quantization is evaluated using representative calibration data.

The calibration configuration is stored in:

`configs/int8_calibration.yaml`

### P20 Pruning

A P20-pruned YOLO11-S model is generated and exported for deployment.

Pruning scripts are stored under:

`optimization/pruning/`

The selected optimized model formats are stored under:

`models/`

## 7. Optimization Evaluation

Optimized models are evaluated using both accuracy and runtime metrics.

Evaluation scripts are stored under:

`optimization/evaluation/`

The comparison includes, where available:

- Precision
- Recall
- mAP@50
- mAP@50-95
- Model size
- Inference latency
- End-to-end FPS

A confidence-threshold sweep is also used to examine the effect of deployment confidence settings. Standard mAP values remain based on the validation procedure rather than being selected from the threshold sweep.

## 8. Jetson Deployment

The Jetson deployment stage covers preparation of the edge platform itself, including:

- microSD system preparation
- Jetson first-boot configuration
- Python and inference environment setup
- CUDA and TensorRT runtime verification
- PyTorch, Ultralytics, and OpenCV setup
- USB camera verification
- Transfer of model and source files
- TensorRT model generation and runtime verification

Deployment documentation is stored in:

`deployment/README.md`

## 9. Final Monitoring Application

After the Jetson environment and optimized models have been verified, the final application combines:

- YOLO fire and smoke object detection
- MobileNetV3-Small scene-level classification
- Real-time camera input
- Dynamic alarm logic
- Runtime monitoring
- Result logging
- Web-based visualization

The final application is stored under:

`application/`

## 10. Experimental Results

Selected result summaries and figures are stored under:

`results/`

The results are intended to document the main conclusions of the project rather than every raw runtime output.

Raw benchmark videos, temporary runtime folders, cache files, and the complete dataset are excluded from version control.
