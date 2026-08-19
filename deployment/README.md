# Jetson Edge Deployment

This directory documents the hardware and software deployment procedure used to run the fire and smoke detection models on the NVIDIA Jetson edge platform.

The deployment stage is separated from the final monitoring application. It focuses on preparing the Jetson hardware, operating system, inference environment, camera, model files, and TensorRT runtime required for real-time execution.

## Hardware Platform

The edge deployment platform used in this project was an NVIDIA Jetson Orin Nano.

The main hardware components included:

- NVIDIA Jetson Orin Nano
- microSD storage
- USB camera
- External display
- Keyboard and other required peripherals

## Deployment Workflow

The deployment procedure followed the general workflow below:

```text
Jetson Orin Nano
        ↓
microSD System Image Preparation
        ↓
Jetson System Installation and First Boot
        ↓
System and Python Environment Configuration
        ↓
CUDA and TensorRT Runtime Verification
        ↓
PyTorch / Ultralytics / OpenCV Setup
        ↓
USB Camera Configuration
        ↓
Source Code and Model Transfer
        ↓
YOLO Runtime Testing
        ↓
TensorRT FP16 / INT8 Model Export
        ↓
Optimized Model Runtime Verification
        ↓
Final Application Execution