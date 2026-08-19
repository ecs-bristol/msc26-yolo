\# Final Fire and Smoke Monitoring Application



This directory contains the final real-time fire and smoke monitoring application developed for the Jetson edge platform.



The application integrates the optimized YOLO detector with a MobileNetV3-Small scene classifier and a real-time alarm and visualization system.



\## Main Application



The main program is:



`jetson\_fire\_alarm.py`



\## System Components



The final application integrates:



\- YOLO-based fire and smoke object detection

\- MobileNetV3-Small scene-level classification

\- Real-time USB camera input

\- Dynamic fire alarm logic

\- Dynamic smoke alarm logic

\- Consecutive-frame alarm confirmation

\- Safe-frame alarm clearing

\- Real-time FPS monitoring

\- YOLO inference latency monitoring

\- Scene-classification latency monitoring

\- Runtime result logging

\- Alarm screenshots

\- CSV and JSON experiment outputs

\- Web-based visualization interface



\## Model Configuration



The YOLO detector and MobileNetV3 scene classifier can be configured independently.



The scene-classification component can be enabled or disabled during runtime, allowing comparison between:



\- YOLO-only detection

\- YOLO + MobileNetV3 scene-assisted detection



The final optimized YOLO models and MobileNetV3 models are stored under:



`models/`



\## Deployment



The hardware and software environment preparation for the NVIDIA Jetson platform is documented separately under:



`deployment/README.md`

