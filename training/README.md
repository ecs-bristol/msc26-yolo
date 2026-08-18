# Training

This directory contains the Google Colab notebooks used to train the four detector variants evaluated in the project.

## Notebooks

| Notebook | Model | Formal Training Setting |
|---|---|---|
| `yolov8_training.ipynb` | YOLOv8-S | 640 px, 100 epochs, batch 16, seed 42 |
| `yolov9_training.ipynb` | YOLOv9-S | 640 px, 100 epochs, batch 8, seed 42 |
| `yolo11_training.ipynb` | YOLO11-S | 640 px, 100 epochs, batch 16, seed 42 |
| `yolo26_training.ipynb` | YOLO26-S | 640 px, 100 epochs, batch 16, seed 42 |

The notebooks cover dataset preparation, model loading, training, validation, and saving experiment outputs.

## Dataset

The notebooks use the Kaggle dataset identifier:

```text
sayedgamal99/smoke-fire-detection-yolo
```

Dataset paths inside Colab may differ from local paths. The repository-level dataset configuration is stored under `../configs/`.

## Expected Training Outputs

Typical Ultralytics training outputs include:

```text
weights/best.pt
weights/last.pt
results.csv
results.png
confusion_matrix.png
PR_curve.png
F1_curve.png
args.yaml
```

Large weight files are not stored in normal Git history. Curated summary tables and selected figures should be copied to `../results/model_comparison/`.

## Reproducibility

For every formal training run, retain the following information:

- Model variant
- Dataset configuration
- Image size
- Number of epochs
- Batch size
- Random seed
- Ultralytics version
- PyTorch version
- CUDA / GPU information
- `args.yaml`
- Final validation metrics
