# Jetson Deployment Application

This directory contains the final real-time fire and smoke monitoring application for the Jetson platform.

## Main Application

```text
jetson_fire_alarm.py
```

The application combines object detection, optional scene-level classification, dynamic alarm logic, runtime monitoring, and a browser-based user interface.

## Main Features

- Runtime discovery of available YOLO detector files
- Safe switching between detector variants
- Support for PyTorch and TensorRT detector models through Ultralytics
- Optional MobileNetV3-Small scene classifier
- Scene assistance ON/OFF control
- Configurable scene-classification interval
- Fire / smoke warning and alarm states
- Alarm persistence and safe-frame clearing logic
- Live MJPEG camera stream
- FPS and latency tracking
- Memory monitoring where available
- Alarm event history
- Alarm screenshots
- CSV and JSON output
- Optional annotated video recording

## Scene Classifier

The scene classifier produces three scene probabilities:

```text
default
fire
smoke
```

Supported scene model formats include PyTorch checkpoints, TorchScript, and ONNX variants handled by the application loader.

## Dynamic Alarm Logic

The default configuration uses separate warning/alarm thresholds for fire and smoke. Consecutive-frame logic is applied to reduce unstable single-frame alarms.

The current default alarm timing configuration is:

```text
Fire alarm:  3 consecutive qualifying frames
Smoke alarm: 5 consecutive qualifying frames
Clear state: 10 safe frames
```

These values are deployment parameters and can be adjusted in the application configuration if a different response/sensitivity trade-off is required.

## Model Directory

Place the required detector and scene-classifier files under:

```text
models/
```

The application scans the model directory at runtime.

## Run

```bash
python jetson_fire_alarm.py
```

Then open a browser on the Jetson device:

```text
http://127.0.0.1:5000
```

If the interface is accessed from another device on the same local network, use the Jetson device's local IP address with port 5000.

## Outputs

Runtime outputs are stored under an application results directory and may include:

- `frames.csv`
- `alarm_events.csv`
- `summary.json`
- alarm screenshots
- annotated video
- master benchmark summary

For the GitHub repository, only selected result tables, screenshots, and plots should be copied to `../results/final_system/`.
