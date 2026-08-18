# Reproducibility Notes

## Training Runs

For every formal training run, preserve:

```text
best.pt
last.pt
results.csv
args.yaml
experiment summary / environment information
selected validation plots
```

Only lightweight summaries and selected figures need to be committed to GitHub.

## Dataset Configuration

Keep the exact YAML file used for each experiment. If a dataset path is modified for Colab or Jetson use, preserve the modified YAML separately instead of silently replacing the original file.

## Model Naming

Use stable descriptive names, for example:

```text
yolov8s_best.pt
yolov9s_best.pt
yolo11s_best.pt
yolo26s_best.pt
yolo11s_best_fp16.engine
yolo11s_best_int8.engine
yolo11s_p20_best.pt
yolo11s_p20_best_fp16.engine
```

## Runtime Comparisons

When comparing models, keep benchmark conditions fixed. Record:

- Input source
- Scenario ID
- Image size
- Confidence threshold
- IoU threshold
- Device
- Duration
- Model path
- Model size
- Software versions

## TensorRT

TensorRT engines should be generated on the target Jetson environment whenever possible. Engine compatibility depends on the TensorRT/CUDA/platform environment.

## INT8 Calibration

INT8 calibration requires representative images. Preserve the calibration dataset definition and the exact calibration YAML used for export.

After INT8 export, run the labelled validation procedure again before reporting accuracy or selecting a final confidence threshold.

## Pruned Model Loading

The P20 pruned checkpoint depends on the pruning-friendly module definition in `pruned_modules.py`. Keep this file with the pruning scripts so the checkpoint can be deserialized correctly.

## Curated Repository Results

Generated runtime folders can be large and repetitive. Keep the raw data locally, but commit only the final summary CSV files and the figures that are used in the report or project discussion.
