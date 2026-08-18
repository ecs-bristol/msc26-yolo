# YOLO11-S Edge Optimization

YOLO11-S is used as the primary detector for the edge-optimization stage after the four-model comparison.

This directory contains TensorRT export, pruning, runtime benchmarking, and accuracy-evaluation utilities.

## Directory Layout

```text
optimization/
├── tensorrt/
│   ├── export_fp16_tensorrt.py
│   └── export_int8_tensorrt.py
├── pruning/
│   ├── pruned_modules.py
│   ├── run_p20.py
│   └── export_pruned_tensorrt.py
├── benchmark/
│   ├── run_yolo11_baseline.py
│   └── run_yolo11_optimized.py
└── evaluation/
    ├── validate_model.py
    ├── threshold_sweep.py
    ├── compare_summaries.py
    └── make_optimization_table.py
```

## 1. Baseline

The trained YOLO11-S PyTorch model is used as the reference model for runtime and validation comparison.

Typical model path:

```text
models/yolo11s_best.pt
```

## 2. TensorRT FP16

FP16 export is performed on the target Jetson environment.

```bash
python tensorrt/export_fp16_tensorrt.py \
  --model ../models/yolo11s_best.pt \
  --imgsz 640 \
  --batch 1 \
  --output ../models/yolo11s_best_fp16.engine
```

The resulting TensorRT engine should be benchmarked using the same input conditions as the baseline.

## 3. TensorRT INT8

INT8 export requires representative calibration data.

```bash
python tensorrt/export_int8_tensorrt.py \
  --model ../models/yolo11s_best.pt \
  --data ../configs/int8_calibration.yaml \
  --imgsz 640 \
  --output ../models/yolo11s_best_int8.engine
```

INT8 deployment should be followed by validation because quantization can change detection behaviour and the preferred confidence threshold.

## 4. Pruning

The pruning workflow contains a custom pruning-friendly module definition (`pruned_modules.py`) required to load the P20 checkpoint.

`run_p20.py` checks the expected pruned architecture before running inference. The pruned checkpoint can subsequently be exported to TensorRT through `export_pruned_tensorrt.py`.

## 5. Runtime Benchmark

Use `benchmark/run_yolo11_baseline.py` for the unoptimized reference model and `benchmark/run_yolo11_optimized.py` for optimized `.pt` or `.engine` variants.

For direct comparison, keep the following conditions fixed:

- Input source
- Image size
- Confidence threshold
- IoU threshold
- Test duration
- Camera/video setup
- Jetson power mode and software environment

## 6. Accuracy Evaluation

`evaluation/validate_model.py` reports:

- Precision
- Recall
- mAP@50
- mAP@50:95

Example:

```bash
python evaluation/validate_model.py \
  --model ../models/yolo11s_best_fp16.engine \
  --data ../configs/data.yaml \
  --output val_fp16.json
```

## 7. Confidence-Threshold Sweep

`evaluation/threshold_sweep.py` evaluates fixed confidence thresholds and reports Precision, Recall, and F1 for deployment threshold selection.

mAP is still obtained from the standard validation procedure and is not selected from the threshold sweep.

## 8. Summary Comparison

`evaluation/compare_summaries.py` compares baseline and optimized runtime summaries.

`evaluation/make_optimization_table.py` combines validation metrics and runtime metrics into one row per model variant for the final optimization table.

Recommended final table fields are:

```text
Variant
Precision
Recall
mAP@50
mAP@50:95
Model size (MB)
Mean inference latency (ms)
P95 inference latency (ms)
End-to-end FPS
```
