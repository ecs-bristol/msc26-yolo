# Model Files

Large model binaries are not tracked in normal Git history by default.

## Trained Detector Weights

Expected trained detector files include:

```text
yolov8s_best.pt
yolov9s_best.pt
yolo11s_best.pt
yolo26s_best.pt
```

## Optimized YOLO11-S Variants

Typical optimized outputs include:

```text
yolo11s_best_fp16.engine
yolo11s_best_int8.engine
yolo11s_p20_best.pt
yolo11s_p20_best_fp16.engine
```

The exact filenames may differ between experiment runs. Update this file if the final model naming scheme changes.

## Scene Classifier

The final application can load a MobileNetV3-based scene classifier from the model directory. Depending on the export path, the scene classifier may be stored as a PyTorch checkpoint, TorchScript model, or ONNX model.

## Why Models Are Excluded from Normal Git

Model files can be large and TensorRT `.engine` files are platform dependent. TensorRT engines are tied to the target runtime environment and should preferably be generated on the deployment Jetson.

If model binaries must be shared through this repository, use Git LFS rather than committing them through normal Git history.
