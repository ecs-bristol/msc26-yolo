# Results

This directory is intended for curated tables and figures used to summarize the experiments.

Raw runtime folders, full videos, complete training runs, and large intermediate outputs should not be committed here.

## Recommended Structure

```text
results/
├── model_comparison/
├── runtime_comparison/
├── optimization/
└── final_system/
```

## 1. Model Comparison

Recommended files:

```text
four_model_metrics.csv
accuracy_comparison.png
model_size_comparison.png
model_complexity_comparison.png
```

The main comparison should summarize YOLOv8-S, YOLOv9-S, YOLO11-S, and YOLO26-S.

## 2. Runtime Comparison

Recommended files:

```text
pc_summary.csv
jetson_summary.csv
pc_vs_jetson_fps.png
pc_vs_jetson_latency.png
scenario_comparison.png
```

The runtime comparison should use consistent scenarios such as fire, smoke, mixed, and negative inputs.

## 3. Optimization

Recommended files:

```text
yolo11_optimization_comparison.csv
optimization_accuracy.png
optimization_latency.png
optimization_fps.png
optimization_model_size.png
threshold_sweep.csv
```

Typical variants may include baseline, FP16, INT8, and pruned/TensorRT models.

## 4. Final System

Recommended files:

```text
final_gui.png
fire_alarm_example.png
smoke_alarm_example.png
scene_assistance_comparison.csv
```

Only screenshots that clearly demonstrate the final system or support the reported analysis should be committed.

## Naming Guidance

Use descriptive filenames. Avoid generic names such as:

```text
test.png
result2.csv
final_final.png
```

Prefer names that identify the experiment and metric directly.
