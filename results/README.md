# Experimental Results

This directory contains selected experimental results from the model comparison, runtime benchmarking, edge optimization, and final fire and smoke monitoring system.

Only representative summary files and key comparison figures are included. Raw per-frame logs, repeated run directories, benchmark videos, and temporary outputs are excluded to keep the repository concise.

## Directory Structure

```text
results/
├── model_comparison/
├── runtime_comparison/
├── optimization/
├── final_system/
└── README.md
```

---

## 1. Model Comparison

Directory:

`model_comparison/`

This directory contains the main results used to compare the four trained object detection models:

- YOLOv8-S
- YOLOv9-S
- YOLO11-S
- YOLO26-S

The model comparison is based primarily on evaluation using the labelled validation dataset.

The main accuracy-related metrics include:

- Precision
- Recall
- F1 score
- mAP@50
- mAP@50-95
- Model size
- Model complexity

The selected summary files include:

- `four_model_metrics.csv`
- `four_model_evaluation_summary.xlsx`

These results are used together with runtime performance to support the selection of YOLO11-S for subsequent edge optimization.

The four benchmark videos are not used to calculate mAP, precision, or recall. These accuracy metrics are obtained from evaluation against labelled validation data.

---

## 2. Runtime Comparison

Directory:

`runtime_comparison/`

This directory contains runtime benchmarking results obtained by running the four YOLO models on PC and NVIDIA Jetson platforms.

Four representative video scenarios were used consistently:

- Fire
- Smoke
- Mixed fire and smoke
- Negative scene

These video experiments are primarily intended to evaluate runtime and deployment behaviour rather than formal detection accuracy.

The main runtime metrics include:

- End-to-end FPS
- Inference latency
- Pre-processing latency
- Post-processing latency
- Model size
- Detection behaviour in representative scenes

Selected files include:

- `pc_all_runs_summary.csv`
- `jetson_all_runs_summary.csv`
- `pc_avg_fps_comparison.png`
- `pc_inference_latency_comparison.png`
- `jetson_video_detection_behaviour.png`

The same representative video scenarios were used across models to support consistent runtime comparison.

Detection counts and detection-frame behaviour from these videos are treated as scenario-based observations and are not used as substitutes for ground-truth accuracy metrics.

---

## 3. Optimization Results

Directory:

`optimization/`

This directory contains selected results from the YOLO11-S edge optimization experiments.

The evaluated optimization methods include:

- TensorRT FP16
- TensorRT INT8
- P20 pruning
- P20-pruned TensorRT deployment

Selected files include:

- `fp16_summary.csv`
- `fp16_int8_comparison.csv`
- `pruning_comparison.csv`
- `p20_fp16_jetson_summary.txt`

The optimization experiments evaluate the trade-off between detection performance, model size, inference latency, and runtime throughput.

### FP16 and INT8

`fp16_summary.csv` contains the selected FP16 runtime results.

`fp16_int8_comparison.csv` contains runtime results used to compare FP16 and INT8 deployment behaviour on the Jetson platform.

### Pruning

`pruning_comparison.csv` contains the comparison of the baseline and pruned YOLO11-S variants.

The pruning experiments include different pruning levels, with the P20 model selected for subsequent Jetson deployment experiments.

### P20 + TensorRT FP16

`p20_fp16_jetson_summary.txt` records the Jetson benchmark of the P20-pruned YOLO11-S TensorRT FP16 model, including preprocessing, inference, postprocessing, total processing latency, and FPS measurements.

Inference-only FPS and end-to-end processing FPS are treated separately when interpreting runtime performance.

---

## 4. Final System

Directory:

`final_system/`

This directory contains the selected summary result from the final Jetson fire and smoke monitoring application.

The final system integrates:

- YOLO fire and smoke detection
- MobileNetV3-Small scene classification
- Dynamic alarm logic
- Real-time Jetson execution
- Runtime monitoring
- Result logging
- Web-based visualization

The selected result file is:

- `master_benchmark.csv`

The original application experiments generated additional per-run files such as frame logs, alarm-event logs, JSON summaries, alarm screenshots, and annotated videos.

These raw outputs are retained separately and are not included in this repository because `master_benchmark.csv` provides the consolidated summary required for reviewing the final system experiments.

---

## Evaluation Strategy

The project uses two complementary forms of evaluation.

### Detection Performance

Detection accuracy is evaluated using labelled validation data.

Metrics include:

- Precision
- Recall
- mAP@50
- mAP@50-95

### Runtime and Scenario-Based Evaluation

Representative fire, smoke, mixed, and negative videos are used to evaluate deployment behaviour.

Metrics include:

- FPS
- Inference latency
- End-to-end processing latency
- Runtime stability
- Scenario-based detection behaviour

This separation ensures that runtime video tests are not interpreted as ground-truth accuracy measurements.

---

## Raw Experimental Outputs

The repository intentionally excludes large or repetitive raw outputs such as:

- Per-frame CSV logs
- Per-detection logs
- Repeated timestamped run directories
- Annotated benchmark videos
- Cache files
- Temporary TensorRT outputs
- Full raw experiment backups

The selected files in this directory preserve the principal results required to understand and reproduce the experimental conclusions.
