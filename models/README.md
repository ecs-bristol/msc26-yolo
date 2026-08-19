# Model Files

This directory contains the trained and optimized model files used in the fire and smoke detection experiments.

## Baseline Detection Models

The following four YOLO models were trained on the same fire and smoke detection dataset and used in the initial model comparison.

| Model file | Description |
|---|---|
| `yolov8s_best.pt` | Trained YOLOv8-S fire and smoke detector |
| `yolov9s_best.pt` | Trained YOLOv9-S fire and smoke detector |
| `yolo11s_best.pt` | Trained YOLO11-S fire and smoke detector |
| `yolo26s_best.pt` | Trained YOLO26-S fire and smoke detector |

Following the comparative evaluation, YOLO11-S was selected for further edge optimization and Jetson deployment.

## Pruned YOLO11-S Models

| Model file | Description |
|---|---|
| `yolo11s_p20_best.pt` | P20-pruned YOLO11-S PyTorch model |
| `yolo11s_p20_best.onnx` | ONNX export of the P20-pruned YOLO11-S model |
| `yolo11s_p20_best_fp16.engine` | TensorRT FP16 engine generated from the P20-pruned model |
| `yolo11s_p20_best_int8.engine` | TensorRT INT8 engine generated from the P20-pruned model |

The TensorRT engines were generated for the Jetson deployment environment used in this project. Compatibility may depend on the JetPack, CUDA, TensorRT, and GPU environment, and regeneration may be required on a different platform.

## Scene Classification Models

MobileNetV3-Small was used as an auxiliary scene-level classifier in the final fire and smoke alarm system.

| Model file | Description |
|---|---|
| `mobilenetv3_scene.pt` | PyTorch MobileNetV3 scene classifier |
| `mobilenetv3_scene.onnx` | ONNX export of the MobileNetV3 scene classifier |

The scene classifier provides additional fire/smoke scene-level information and can be enabled or disabled in the final Jetson deployment system.

## Model Usage

The four baseline YOLO models are used by the evaluation scripts under:

`evaluation/`

YOLO11-S optimization and validation scripts are located under:

`optimization/`

The final YOLO and MobileNetV3 integrated deployment system is located under:

`deployment/`