# Experimental Setup

## Objective

The experimental workflow is designed to compare multiple YOLO detector variants and then evaluate edge-optimization approaches for real-time fire and smoke detection.

## Detector Training

The formal model-training settings are:

| Model | Image Size | Epochs | Batch Size | Seed |
|---|---:|---:|---:|---:|
| YOLOv8-S | 640 | 100 | 16 | 42 |
| YOLOv9-S | 640 | 100 | 8 | 42 |
| YOLO11-S | 640 | 100 | 16 | 42 |
| YOLO26-S | 640 | 100 | 16 | 42 |

The same fire/smoke dataset is used across the four detector experiments.

## Detection Metrics

The model-level comparison uses standard object-detection metrics where available:

- Precision
- Recall
- mAP@50
- mAP@50:95

Additional deployment-oriented comparisons include model size, model complexity, inference latency, and end-to-end FPS.

## Runtime Benchmark Protocol

The PC and Jetson scripts are designed around a common benchmark structure:

1. Load the selected model.
2. Open a camera or test video.
3. Perform warm-up inference.
4. Start a formal timed run.
5. Record per-frame latency and detections.
6. Save a run-level summary.
7. Generate selected plots and tables.

For direct comparisons, use the same:

- Test source
- Input resolution
- Confidence threshold
- IoU threshold
- Test duration
- Scenario definition

## Scenario Labels

Recommended runtime scenario identifiers are:

```text
fire01
smoke01
mixed01
negative01
```

## Edge Platform

The edge deployment target is NVIDIA Jetson Orin Nano.

Jetson experiments should record the active software stack and device information for each benchmark run, including PyTorch, CUDA, Ultralytics, OpenCV, and device model information where available.

## Optimization Stage

YOLO11-S is used as the primary model for further edge optimization.

The implemented optimization workflow covers:

- TensorRT FP16
- TensorRT INT8 with representative calibration data
- P20 pruning workflow
- TensorRT export of the pruned model
- Accuracy re-validation
- Confidence-threshold sweep
- Runtime re-benchmarking

## Accuracy vs Runtime

Accuracy metrics from the labelled validation/test data and runtime metrics from camera/video benchmarking are treated as separate measurements. This avoids interpreting camera detection frequency as a substitute for object-detection accuracy.
