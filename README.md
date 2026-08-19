# Real-Time Fire and Smoke Detection with YOLO and Edge Deployment

This repository contains the training, evaluation, optimization, and edge deployment code for a real-time fire and smoke detection system.

The project compares four YOLO detector variants and then selects YOLO11-S for further optimization and deployment on an NVIDIA Jetson platform. The final system combines YOLO-based object detection with an auxiliary MobileNetV3-Small scene classifier and a dynamic fire/smoke alarm mechanism.

---

## Project Workflow

```text
Fire and Smoke Dataset
        ↓
YOLOv8-S / YOLOv9-S / YOLO11-S / YOLO26-S
        ↓
Training and Validation
        ↓
Four-Model Comparison
        ↓
PC and Jetson Runtime Evaluation
        ↓
YOLO11-S Selected
        ↓
TensorRT FP16 / INT8 Optimization
        ↓
P20 Model Pruning
        ↓
P20 + TensorRT FP16 / INT8
        ↓
MobileNetV3 Scene Assistance
        ↓
Final Jetson Real-Time Alarm System
```

---

## Repository Structure

```text
msc26-yolo/
│
├── configs/
│   ├── data.yaml
│   ├── int8_calibration.yaml
│   └── README.md
│
├── training/
│   ├── yolov8_training.ipynb
│   ├── yolov9_training.ipynb
│   ├── yolo11_training.ipynb
│   ├── yolo26_training.ipynb
│   └── README.md
│
├── evaluation/
│   ├── model_comparison/
│   ├── pc/
│   ├── jetson/
│   └── README.md
│
├── optimization/
│   ├── tensorrt/
│   ├── pruning/
│   ├── benchmark/
│   ├── evaluation/
│   └── README.md
│
├── deployment/
│   └── README.md
│
├── application/
│   ├── jetson_fire_alarm.py
│   └── README.md
│
├── models/
│   ├── yolov8s_best.pt
│   ├── yolov9s_best.pt
│   ├── yolo11s_best.pt
│   ├── yolo26s_best.pt
│   ├── yolo11s_p20_best.pt
│   ├── yolo11s_p20_best.onnx
│   ├── yolo11s_p20_best_fp16.engine
│   ├── yolo11s_p20_best_int8.engine
│   ├── mobilenetv3_scene.pt
│   ├── mobilenetv3_scene.onnx
│   └── README.md
│
├── results/
│   └── README.md
│
├── docs/
│   ├── DATASET.md
│   ├── EXPERIMENTAL_SETUP.md
│   └── REPRODUCIBILITY.md
│
├── requirements.txt
├── requirements_jetson.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Dataset

The project uses a fire and smoke detection dataset obtained from Kaggle.

The dataset contains separate training, validation, and test subsets.

The complete image dataset is not included in this repository. Dataset configuration files are provided under:

```text
configs/
├── data.yaml
└── int8_calibration.yaml
```

`data.yaml` is used for standard model training and validation.

`int8_calibration.yaml` is used for representative calibration during TensorRT INT8 model conversion.

---

## Model Training

Four YOLO detector variants were trained and evaluated:

| Model    | Image Size | Epochs | Batch Size | Seed |
| -------- | ---------: | -----: | ---------: | ---: |
| YOLOv8-S |        640 |    100 |         16 |   42 |
| YOLOv9-S |        640 |    100 |          8 |   42 |
| YOLO11-S |        640 |    100 |         16 |   42 |
| YOLO26-S |        640 |    100 |         16 |   42 |

All models used the same image resolution, number of training epochs, and random seed.

YOLOv9-S was trained using a batch size of 8 in the completed experiment.

The corresponding Google Colab notebooks are available under:

```text
training/
```

Each notebook contains dataset preparation, model loading, training, validation, and experiment output generation.

---

## Four-Model Evaluation

The four trained models were compared using detection accuracy and runtime-related metrics.

The evaluation includes:

* Precision
* Recall
* mAP@50
* mAP@50–95
* Model size
* Inference latency
* End-to-end FPS
* Fire and smoke detection behaviour
* Negative-scene behaviour

The model comparison notebook is located under:

```text
evaluation/model_comparison/
```

Based on the overall balance between detection performance and deployment efficiency, **YOLO11-S was selected for further edge optimization**.

---

## PC and Jetson Benchmarking

Separate runtime benchmark scripts are provided for PC and Jetson environments.

### PC Benchmark

```text
evaluation/pc/
├── run_yolov8.py
├── run_yolov9.py
├── run_yolo11.py
└── run_yolo26.py
```

### Jetson Benchmark

```text
evaluation/jetson/
├── run_jetson_yolov8.py
├── run_jetson_yolov9.py
├── run_jetson_yolo11.py
└── run_jetson_yolo26.py
```

The benchmark scripts record metrics including:

* End-to-end FPS
* Inference latency
* Pre-processing latency
* Post-processing latency
* Detection counts
* Fire detection counts
* Smoke detection counts
* Detection confidence
* Model size
* Runtime environment information

The same fire, smoke, mixed, and negative scenarios were used to support consistent comparison across models and platforms.

---

# YOLO11-S Edge Optimization

After the four-model comparison, YOLO11-S was selected as the main detector for edge optimization.

The optimization stage includes:

1. TensorRT FP16 conversion
2. TensorRT INT8 quantization
3. Model pruning
4. P20-pruned TensorRT deployment
5. Accuracy re-evaluation
6. Confidence-threshold evaluation
7. Runtime comparison

---

## TensorRT FP16

TensorRT FP16 conversion was used to reduce inference latency and improve deployment efficiency on the Jetson platform.

The FP16 export script is located at:

```text
optimization/tensorrt/export_fp16_tensorrt.py
```

The final P20-pruned model was also converted to TensorRT FP16.

The resulting deployment engine is included as:

```text
models/yolo11s_p20_best_fp16.engine
```

---

## TensorRT INT8

INT8 quantization was evaluated to further reduce computational cost and model runtime.

Representative fire and smoke data were used for INT8 calibration.

The INT8 export script is located at:

```text
optimization/tensorrt/export_int8_tensorrt.py
```

The final P20-pruned INT8 TensorRT engine is included as:

```text
models/yolo11s_p20_best_int8.engine
```

Validation is performed after INT8 conversion because quantization can affect detection accuracy and the optimal confidence threshold.

---

## Model Pruning

A P20-pruned YOLO11-S model was generated to reduce model complexity while retaining fire and smoke detection capability.

The pruning-related implementation is located under:

```text
optimization/pruning/
```

Important files include:

```text
pruned_modules.py
run_p20.py
export_pruned_tensorrt.py
```

The resulting pruned models are:

```text
models/yolo11s_p20_best.pt
models/yolo11s_p20_best.onnx
models/yolo11s_p20_best_fp16.engine
models/yolo11s_p20_best_int8.engine
```

---

## Optimization Evaluation

Optimization evaluation scripts are located under:

```text
optimization/evaluation/
```

They include:

```text
validate_model.py
threshold_sweep.py
compare_summaries.py
make_optimization_table.py
```

### Validation

`validate_model.py` evaluates trained or optimized models on the labelled validation dataset and reports:

* Precision
* Recall
* mAP@50
* mAP@50–95

### Confidence Threshold Sweep

`threshold_sweep.py` evaluates multiple confidence thresholds and reports:

* Precision
* Recall
* F1 score
* mAP metrics

This is used to examine suitable deployment confidence settings independently from standard mAP reporting.

### Runtime Comparison

`compare_summaries.py` compares baseline and optimized runtime results, including:

* Mean inference latency
* Latency reduction
* Inference speed-up
* End-to-end FPS
* Model size

### Optimization Summary

`make_optimization_table.py` combines validation and runtime results into a single comparison table for different optimization variants.

---

# MobileNetV3 Scene Assistance

The final system includes a MobileNetV3-Small scene classifier as an auxiliary component.

The classifier performs scene-level classification using three categories:

```text
default
fire
smoke
```

Two formats are included:

```text
models/mobilenetv3_scene.pt
models/mobilenetv3_scene.onnx
```

Scene assistance can be enabled or disabled in the final system.

The scene classifier can also be executed at configurable frame intervals to reduce additional computational cost.

---
# Jetson Edge Deployment

The selected detection models were deployed on an NVIDIA Jetson edge platform.

The deployment procedure included microSD system preparation, Jetson system configuration, Python and inference environment setup, camera configuration, model transfer, TensorRT conversion, and runtime verification.

Detailed hardware and software deployment information is provided under:

`deployment/README.md`

---
# Final Fire and Smoke Monitoring Application

The final real-time fire and smoke alarm application is located at:

```text
application/jetson_fire_alarm.py
```

The system combines:

* YOLO fire and smoke object detection
* MobileNetV3 scene-level classification
* Dynamic fire alarm logic
* Dynamic smoke alarm logic
* Consecutive-frame confirmation
* Safe-frame alarm clearing
* Real-time camera input
* FPS monitoring
* Inference latency monitoring
* Alarm history
* Runtime result logging
* Alarm screenshots
* CSV and JSON experiment outputs
* Web-based visualization

The detection and scene-classification components can be configured independently during deployment.

---

# Model Files

Selected trained and optimized model files are included under the `models/` directory.

| Model File                     | Description                                              |
| ------------------------------ | -------------------------------------------------------- |
| `yolov8s_best.pt`              | Trained YOLOv8-S fire and smoke detector                 |
| `yolov9s_best.pt`              | Trained YOLOv9-S fire and smoke detector                 |
| `yolo11s_best.pt`              | Trained YOLO11-S detector selected for optimization      |
| `yolo26s_best.pt`              | Trained YOLO26-S fire and smoke detector                 |
| `yolo11s_p20_best.pt`          | P20-pruned YOLO11-S PyTorch model                        |
| `yolo11s_p20_best.onnx`        | ONNX export of the P20-pruned YOLO11-S model             |
| `yolo11s_p20_best_fp16.engine` | TensorRT FP16 engine generated from the P20-pruned model |
| `yolo11s_p20_best_int8.engine` | TensorRT INT8 engine generated from the P20-pruned model |
| `mobilenetv3_scene.pt`         | MobileNetV3-Small scene classifier                       |
| `mobilenetv3_scene.onnx`       | ONNX export of the MobileNetV3 scene classifier          |

TensorRT engine files were generated for the Jetson deployment environment used in this project.

Compatibility of `.engine` files depends on the JetPack, CUDA, TensorRT, and GPU environment. Regeneration may therefore be required on a different Jetson or TensorRT environment.

---

# Installation

## PC / Google Colab

Install the required Python dependencies using:

```bash
pip install -r requirements.txt
```

## Jetson

Jetson environments depend on the installed JetPack, CUDA, TensorRT, PyTorch, and OpenCV versions.

See:

```text
requirements_jetson.txt
```

for the main runtime dependencies.

---

# Example Usage

## PC YOLO11-S Benchmark

```bash
python evaluation/pc/run_yolo11.py \
  --model models/yolo11s_best.pt \
  --source 0 \
  --device 0 \
  --conf 0.15 \
  --imgsz 640
```

## Jetson YOLO11-S Benchmark

```bash
python evaluation/jetson/run_jetson_yolo11.py \
  --model models/yolo11s_best.pt \
  --source 0 \
  --device 0 \
  --conf 0.15 \
  --imgsz 640
```

## Optimized YOLO11 Benchmark

```bash
python optimization/benchmark/run_yolo11_optimized.py \
  --model models/yolo11s_p20_best_fp16.engine \
  --source 0 \
  --device 0 \
  --conf 0.15 \
  --imgsz 640
```

## Model Validation

```bash
python optimization/evaluation/validate_model.py \
  --model models/yolo11s_best.pt \
  --data configs/data.yaml \
  --device 0
```

---

# Experimental Results

Experiment results are organized under:

```text
results/
```

The repository structure supports results from:

* Four-model comparison
* PC runtime benchmarking
* Jetson runtime benchmarking
* TensorRT FP16 optimization
* TensorRT INT8 optimization
* P20 pruning
* Validation and confidence-threshold analysis
* Final integrated Jetson system

Raw datasets, large benchmark videos, temporary runtime files, and cache directories are excluded from version control.

---

# Reproducibility

Additional information about the experiment setup and reproduction procedure is provided under:

```text
docs/
├── DATASET.md
├── EXPERIMENTAL_SETUP.md
└── REPRODUCIBILITY.md
```

Model training parameters, deployment scripts, dataset configuration files, selected trained weights, and optimization scripts are included to support reproduction of the main experimental workflow.

---

# License

This repository follows the license provided in the root `LICENSE` file.
