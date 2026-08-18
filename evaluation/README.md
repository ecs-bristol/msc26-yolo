# Evaluation

This directory contains scripts and notebooks used to compare the four trained YOLO detector variants and evaluate runtime behaviour on PC and Jetson platforms.

## Directory Layout

```text
evaluation/
├── model_comparison/
│   └── four_model_comparison.ipynb
├── pc/
│   ├── run_yolov8.py
│   ├── run_yolov9.py
│   ├── run_yolo11.py
│   └── run_yolo26.py
└── jetson/
    ├── run_jetson_yolov8.py
    ├── run_jetson_yolov9.py
    ├── run_jetson_yolo11.py
    └── run_jetson_yolo26.py
```

## Four-Model Comparison

`four_model_comparison.ipynb` is used to consolidate model-level information and compare the trained detector variants.

Recommended comparison metrics include:

- Precision
- Recall
- mAP@50
- mAP@50:95
- Model size
- Parameter count / GFLOPs where available
- Inference speed

## PC Runtime Benchmark

The PC benchmark scripts share a common workflow so that each model is measured using the same runtime logic.

The scripts support:

- Camera index or video path as input
- GPU or CPU selection with `--device`
- Fixed confidence and IoU thresholds
- Fixed inference resolution
- Warm-up inference
- Formal timed benchmark interval
- Per-frame FPS and latency recording
- Detection logging
- Annotated video output
- CSV, JSON, Excel, and figure generation

Example:

```bash
python pc/run_yolov8.py --source 0 --device 0 --conf 0.15 --imgsz 640 --duration 60 --test-id fire01
```

## Jetson Runtime Benchmark

Jetson scripts use a similar measurement pipeline but include Jetson-specific camera opening and hardware information.

Example:

```bash
python jetson/run_jetson_yolo11.py --source 0 --device 0 --conf 0.15 --imgsz 640 --duration 60 --test-id mixed01
```

## Test Scenarios

A consistent set of scenario identifiers should be used across models and platforms. Recommended identifiers are:

```text
fire01
smoke01
mixed01
negative01
```

The same source videos, inference size, confidence threshold, and test duration should be used when making direct model/platform comparisons.

## Curated Outputs

Do not commit every generated runtime folder. Copy only final summary files and figures into `../results/runtime_comparison/`.
