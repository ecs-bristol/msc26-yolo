# -*- coding: utf-8 -*-
"""
YOLOv9-S - Real-time Fire/Smoke Camera Benchmark
Designed for fair PC GPU / Jetson comparison.

Keys:
    S / SPACE : start formal recording/benchmark
    P         : pause/resume during benchmark
    Q / ESC   : finish and save results

Example:
    python run_yolov9.py --source 0 --device 0 --conf 0.15 --duration 60 --test-id fire01
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO

MODEL_LABEL = "YOLOv9-S"
DEFAULT_MODEL = r"models/yolov9s_best.pt"


def parse_source(value):
    value = str(value)
    return int(value) if value.isdigit() else value


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def class_group(name):
    n = str(name).strip().lower()
    if "fire" in n or "flame" in n:
        return "fire"
    if "smoke" in n:
        return "smoke"
    return "other"


def get_gpu_name(device_arg):
    if str(device_arg).lower() == "cpu":
        return "CPU"
    if torch.cuda.is_available():
        try:
            idx = int(str(device_arg).split(",")[0])
        except Exception:
            idx = 0
        return torch.cuda.get_device_name(idx)
    return "CUDA unavailable"


def get_nvidia_smi():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
        return subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL, timeout=3
        ).strip()
    except Exception:
        return ""


def ensure_writer(path, fps, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if writer.isOpened():
        return writer, path

    fallback = path.with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(fallback), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError("Could not open a video writer for MP4 or AVI.")
    return writer, fallback


def draw_text(img, text, xy, scale=0.58, thickness=1):
    cv2.putText(
        img,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (235, 235, 235),
        thickness,
        cv2.LINE_AA,
    )


def build_dashboard(frame, metrics, duration_limit):
    h, w = frame.shape[:2]
    panel_w = 360
    panel = np.full((h, panel_w, 3), 24, dtype=np.uint8)

    y = 34
    draw_text(panel, MODEL_LABEL, (20, y), 0.82, 2)
    y += 33
    draw_text(panel, f"Device: {metrics['device']}", (20, y), 0.52)
    y += 25
    draw_text(panel, f"Test: {metrics['test_id']}", (20, y), 0.52)

    y += 40
    draw_text(panel, f"State: {metrics['state']}", (20, y), 0.62, 2)
    y += 35
    draw_text(panel, f"Elapsed: {metrics['elapsed']:.1f} s", (20, y), 0.58)
    y += 27
    draw_text(panel, f"Current FPS: {metrics['fps']:.2f}", (20, y), 0.58)
    y += 27
    draw_text(panel, f"Average FPS: {metrics['avg_fps']:.2f}", (20, y), 0.58)
    y += 27
    draw_text(panel, f"Inference: {metrics['inference_ms']:.2f} ms", (20, y), 0.58)

    y += 42
    draw_text(panel, f"Detections now: {metrics['detections']}", (20, y), 0.58)
    y += 27
    draw_text(panel, f"Fire now: {metrics['fire_now']}", (20, y), 0.58)
    y += 27
    draw_text(panel, f"Smoke now: {metrics['smoke_now']}", (20, y), 0.58)

    y += 42
    draw_text(panel, "Controls", (20, y), 0.62, 2)
    y += 29
    draw_text(panel, "SPACE/S  Start", (20, y), 0.52)
    y += 24
    draw_text(panel, "P        Pause/resume", (20, y), 0.52)
    y += 24
    draw_text(panel, "Q/ESC    Save & quit", (20, y), 0.52)

    if duration_limit > 0:
        progress = min(max(metrics["elapsed"] / duration_limit, 0.0), 1.0)
        bx0, by0, bx1, by1 = 20, h - 48, panel_w - 20, h - 28
        cv2.rectangle(panel, (bx0, by0), (bx1, by1), (90, 90, 90), 1)
        fill_x = int(bx0 + (bx1 - bx0) * progress)
        cv2.rectangle(
            panel,
            (bx0 + 2, by0 + 2),
            (max(bx0 + 2, fill_x), by1 - 2),
            (200, 200, 200),
            -1,
        )

    return np.hstack([frame, panel])


def save_plots(frames_df, det_df, run_dir):
    if frames_df.empty:
        return

    fig = plt.figure(figsize=(9, 4.8))
    plt.plot(frames_df["elapsed_s"], frames_df["fps_e2e"])
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("End-to-end FPS")
    plt.title(f"{MODEL_LABEL} - FPS over time")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(run_dir / "01_fps_over_time.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 4.8))
    plt.plot(frames_df["elapsed_s"], frames_df["inference_ms"])
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Inference latency (ms)")
    plt.title(f"{MODEL_LABEL} - Inference latency over time")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(run_dir / "02_inference_latency_over_time.png", dpi=180)
    plt.close(fig)

    counts = {
        "Fire detections": int(frames_df["fire_count"].sum()),
        "Smoke detections": int(frames_df["smoke_count"].sum()),
        "Other detections": int(frames_df["other_count"].sum()),
    }
    fig = plt.figure(figsize=(7, 4.8))
    plt.bar(list(counts.keys()), list(counts.values()))
    plt.ylabel("Detection boxes across frames")
    plt.title(f"{MODEL_LABEL} - Detection counts")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(run_dir / "03_detection_counts.png", dpi=180)
    plt.close(fig)

    if not det_df.empty and "confidence" in det_df:
        fig = plt.figure(figsize=(8, 4.8))
        plt.hist(det_df["confidence"].dropna(), bins=20)
        plt.xlabel("Confidence")
        plt.ylabel("Number of detections")
        plt.title(f"{MODEL_LABEL} - Detection confidence distribution")
        plt.tight_layout()
        fig.savefig(run_dir / "04_confidence_distribution.png", dpi=180)
        plt.close(fig)


def update_master_summary(summary_row, results_root):
    results_root.mkdir(parents=True, exist_ok=True)
    master_csv = results_root / "all_runs_summary.csv"
    master_xlsx = results_root / "all_runs_summary.xlsx"

    new_df = pd.DataFrame([summary_row])
    if master_csv.exists():
        try:
            old = pd.read_csv(master_csv)
            all_df = pd.concat([old, new_df], ignore_index=True)
        except Exception:
            all_df = new_df
    else:
        all_df = new_df

    if "run_id" in all_df.columns:
        all_df = all_df.drop_duplicates(subset=["run_id"], keep="last")

    all_df.to_csv(master_csv, index=False, encoding="utf-8-sig")
    try:
        all_df.to_excel(master_xlsx, index=False)
    except Exception as exc:
        print(f"[WARN] Excel export skipped: {exc}")

    valid = all_df.copy()
    for col in ["avg_fps_e2e", "mean_inference_ms"]:
        valid[col] = pd.to_numeric(valid[col], errors="coerce")

    grouped = (
        valid.groupby(["platform_tag", "model"], dropna=False)[
            ["avg_fps_e2e", "mean_inference_ms"]
        ]
        .mean()
        .reset_index()
    )
    if len(grouped) >= 2:
        labels = (
            grouped["platform_tag"].astype(str)
            + " | "
            + grouped["model"].astype(str)
        )

        fig = plt.figure(figsize=(10, 5.2))
        plt.bar(labels, grouped["avg_fps_e2e"])
        plt.ylabel("Mean end-to-end FPS")
        plt.title("Model / platform FPS comparison")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        fig.savefig(results_root / "comparison_avg_fps.png", dpi=180)
        plt.close(fig)

        fig = plt.figure(figsize=(10, 5.2))
        plt.bar(labels, grouped["mean_inference_ms"])
        plt.ylabel("Mean inference latency (ms)")
        plt.title("Model / platform latency comparison")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        fig.savefig(
            results_root / "comparison_inference_latency.png", dpi=180
        )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=f"{MODEL_LABEL} camera benchmark"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Path to this model's best.pt"
    )
    parser.add_argument(
        "--source", default="0", help="Camera index (0/1) or video file path"
    )
    parser.add_argument(
        "--device", default="0", help="CUDA device, e.g. 0, or cpu"
    )
    parser.add_argument(
        "--conf", type=float, default=0.15, help="Confidence threshold"
    )
    parser.add_argument(
        "--iou", type=float, default=0.70, help="IoU threshold"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Formal test duration in seconds; 0=no limit",
    )
    parser.add_argument(
        "--test-id",
        default="test01",
        help="Scenario ID: negative01/fire01/smoke01/mixed01",
    )
    parser.add_argument(
        "--platform-tag",
        default="PC_GPU",
        help="PC_GPU or JETSON for later comparison",
    )
    parser.add_argument(
        "--results-root",
        default="results_camera",
        help="Root folder for results",
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Requested camera width"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Requested camera height"
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=30.0,
        help="Requested camera FPS",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warm-up inferences excluded from metrics",
    )
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Do not save annotated video",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print("\n[ERROR] Model file not found:")
        print(f"        {model_path.resolve()}")
        print(
            "\nPut best.pt at the expected path, "
            "or pass --model YOUR_PATH\\best.pt"
        )
        sys.exit(2)

    if str(args.device).lower() != "cpu" and not torch.cuda.is_available():
        print(
            "[ERROR] GPU requested but torch.cuda.is_available() is False."
        )
        print(
            "Check the PyCharm interpreter and PyTorch CUDA installation."
        )
        sys.exit(3)

    source = parse_source(args.source)
    is_camera = isinstance(source, int)

    print(f"\n=== {MODEL_LABEL} ===")
    print(f"Model: {model_path.resolve()}")
    print(f"Device argument: {args.device}")
    print(f"GPU: {get_gpu_name(args.device)}")
    print(
        f"Torch: {torch.__version__} | CUDA runtime: {torch.version.cuda}"
    )
    print(f"Ultralytics: {ultralytics.__version__}")
    print("Loading model...")

    model = YOLO(str(model_path))

    if is_camera and platform.system().lower().startswith("win"):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    if is_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.camera_fps)

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        raise RuntimeError(
            "Camera/video opened, but no frame could be read."
        )

    actual_h, actual_w = first_frame.shape[:2]
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    if reported_fps <= 1 or reported_fps > 240:
        reported_fps = args.camera_fps

    print(
        f"Capture: {actual_w}x{actual_h} @ reported "
        f"{reported_fps:.2f} FPS"
    )
    print(
        f"Warming up model for {args.warmup} inference(s)..."
    )

    for _ in range(max(0, args.warmup)):
        _ = model.predict(
            first_frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        f"{args.platform_tag}_{MODEL_LABEL}_{args.test_id}_{timestamp}"
    )
    run_dir = (
        Path(args.results_root)
        / args.platform_tag
        / MODEL_LABEL
        / args.test_id
        / timestamp
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    video_out_path = run_dir / "annotated.mp4"

    frames_rows = []
    det_rows = []
    formal_started = False
    paused = False
    start_time = None
    last_frame_end = None
    processed_frames = 0
    total_detections = 0
    fire_detections = 0
    smoke_detections = 0
    fire_frames = 0
    smoke_frames = 0
    any_detection_frames = 0

    print("\nREADY.")
    print("1) Put the phone at the fixed position.")
    print("2) Pause the test video at the exact start frame.")
    print("3) Press SPACE or S in the camera window to START.")
    print("4) Start the phone video immediately.")
    print("The script auto-stops at --duration.\n")

    current_frame = first_frame

    while True:
        if current_frame is None:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            frame = current_frame
            current_frame = None

        if paused and formal_started:
            metrics = {
                "device": get_gpu_name(args.device),
                "test_id": args.test_id,
                "state": "PAUSED",
                "elapsed": max(0.0, time.perf_counter() - start_time),
                "fps": 0.0,
                "avg_fps": (
                    0.0
                    if not frames_rows
                    else float(
                        np.mean(
                            [
                                r["fps_e2e"]
                                for r in frames_rows
                                if r["fps_e2e"] > 0
                            ]
                        )
                    )
                ),
                "inference_ms": 0.0,
                "detections": 0,
                "fire_now": 0,
                "smoke_now": 0,
            }
            dashboard = build_dashboard(
                frame.copy(), metrics, args.duration
            )
            cv2.imshow(
                f"{MODEL_LABEL} Fire/Smoke Benchmark", dashboard
            )
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("p"), ord("P")):
                paused = False
            elif key in (ord("q"), ord("Q"), 27):
                break
            continue

        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        result = results[0]
        annotated = result.plot()

        speed = getattr(result, "speed", {}) or {}
        preprocess_ms = safe_float(speed.get("preprocess", 0.0))
        inference_ms = safe_float(speed.get("inference", 0.0))
        postprocess_ms = safe_float(speed.get("postprocess", 0.0))

        frame_det_count = 0
        fire_now = 0
        smoke_now = 0
        other_now = 0
        confidences = []
        class_counter = Counter()

        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy_list = boxes.xyxy.detach().cpu().numpy()
            conf_list = boxes.conf.detach().cpu().numpy()
            cls_list = boxes.cls.detach().cpu().numpy().astype(int)
            names = result.names

            for xyxy, confv, cls_id in zip(
                xyxy_list, conf_list, cls_list
            ):
                if isinstance(names, dict):
                    class_name = str(names.get(int(cls_id), cls_id))
                else:
                    class_name = str(names[int(cls_id)])
                grp = class_group(class_name)
                frame_det_count += 1
                confidences.append(float(confv))
                class_counter[class_name] += 1

                if grp == "fire":
                    fire_now += 1
                elif grp == "smoke":
                    smoke_now += 1
                else:
                    other_now += 1

                if formal_started:
                    det_rows.append(
                        {
                            "run_id": run_id,
                            "frame_idx": processed_frames + 1,
                            "elapsed_s": max(
                                0.0,
                                time.perf_counter() - start_time,
                            ),
                            "class_id": int(cls_id),
                            "class_name": class_name,
                            "group": grp,
                            "confidence": float(confv),
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "x2": float(xyxy[2]),
                            "y2": float(xyxy[3]),
                        }
                    )

        now = time.perf_counter()
        if last_frame_end is None:
            fps_e2e = 0.0
        else:
            dt = now - last_frame_end
            fps_e2e = 1.0 / dt if dt > 0 else 0.0
        last_frame_end = now

        elapsed = (
            0.0
            if start_time is None
            else max(0.0, now - start_time)
        )

        if formal_started:
            processed_frames += 1
            total_detections += frame_det_count
            fire_detections += fire_now
            smoke_detections += smoke_now

            if frame_det_count > 0:
                any_detection_frames += 1
            if fire_now > 0:
                fire_frames += 1
            if smoke_now > 0:
                smoke_frames += 1

            frames_rows.append(
                {
                    "run_id": run_id,
                    "model": MODEL_LABEL,
                    "platform_tag": args.platform_tag,
                    "test_id": args.test_id,
                    "frame_idx": processed_frames,
                    "elapsed_s": elapsed,
                    "fps_e2e": fps_e2e,
                    "preprocess_ms": preprocess_ms,
                    "inference_ms": inference_ms,
                    "postprocess_ms": postprocess_ms,
                    "pipeline_model_ms": (
                        preprocess_ms
                        + inference_ms
                        + postprocess_ms
                    ),
                    "detection_count": frame_det_count,
                    "fire_count": fire_now,
                    "smoke_count": smoke_now,
                    "other_count": other_now,
                    "mean_confidence": (
                        float(np.mean(confidences))
                        if confidences
                        else np.nan
                    ),
                    "max_confidence": (
                        float(np.max(confidences))
                        if confidences
                        else np.nan
                    ),
                    "class_counts_json": json.dumps(
                        dict(class_counter),
                        ensure_ascii=False,
                    ),
                }
            )

            if not args.no_save_video:
                if writer is None:
                    writer, video_out_path = ensure_writer(
                        video_out_path,
                        reported_fps,
                        (annotated.shape[1], annotated.shape[0]),
                    )
                writer.write(annotated)

        valid_live_fps = [
            r["fps_e2e"]
            for r in frames_rows
            if r["fps_e2e"] > 0
        ]
        avg_fps = (
            float(np.mean(valid_live_fps))
            if valid_live_fps
            else 0.0
        )

        state = (
            "RUNNING"
            if formal_started
            else "READY - press SPACE/S"
        )
        metrics = {
            "device": get_gpu_name(args.device),
            "test_id": args.test_id,
            "state": state,
            "elapsed": elapsed,
            "fps": fps_e2e,
            "avg_fps": avg_fps,
            "inference_ms": inference_ms,
            "detections": frame_det_count,
            "fire_now": fire_now,
            "smoke_now": smoke_now,
        }
        dashboard = build_dashboard(
            annotated, metrics, args.duration
        )
        cv2.imshow(
            f"{MODEL_LABEL} Fire/Smoke Benchmark", dashboard
        )

        if (
            formal_started
            and args.duration > 0
            and elapsed >= args.duration
        ):
            print(
                f"Reached {args.duration:.1f} seconds. "
                "Finishing test..."
            )
            break

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif (
            key in (ord(" "), ord("s"), ord("S"))
            and not formal_started
        ):
            formal_started = True
            start_time = time.perf_counter()
            last_frame_end = None
            print("FORMAL TEST STARTED.")
        elif (
            key in (ord("p"), ord("P"))
            and formal_started
        ):
            paused = True

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if not formal_started or not frames_rows:
        print(
            "No formal test data recorded. "
            "Nothing to summarize."
        )
        return

    frames_df = pd.DataFrame(frames_rows)
    det_df = pd.DataFrame(det_rows)

    frames_df.to_csv(
        run_dir / "frames.csv",
        index=False,
        encoding="utf-8-sig",
    )
    det_df.to_csv(
        run_dir / "detections.csv",
        index=False,
        encoding="utf-8-sig",
    )

    valid_fps = frames_df.loc[
        frames_df["fps_e2e"] > 0, "fps_e2e"
    ]
    valid_inf = frames_df["inference_ms"].dropna()
    all_conf = (
        det_df["confidence"].dropna()
        if not det_df.empty
        else pd.Series(dtype=float)
    )

    model_size_mb = (
        model_path.stat().st_size / (1024 * 1024)
        if model_path.exists()
        else np.nan
    )

    total_elapsed = float(frames_df["elapsed_s"].max())
    avg_fps_e2e = (
        float(valid_fps.mean())
        if len(valid_fps)
        else 0.0
    )

    summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "platform_tag": args.platform_tag,
        "model": MODEL_LABEL,
        "model_path": str(model_path.resolve()),
        "model_size_mb": round(model_size_mb, 3),
        "test_id": args.test_id,
        "source": str(args.source),
        "device_arg": str(args.device),
        "device_name": get_gpu_name(args.device),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda),
        "ultralytics_version": ultralytics.__version__,
        "nvidia_smi": get_nvidia_smi(),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "capture_width": actual_w,
        "capture_height": actual_h,
        "capture_reported_fps": reported_fps,
        "duration_s": round(total_elapsed, 3),
        "processed_frames": int(processed_frames),
        "avg_fps_e2e": round(avg_fps_e2e, 4),
        "median_fps_e2e": (
            round(float(valid_fps.median()), 4)
            if len(valid_fps)
            else 0.0
        ),
        "p05_fps_e2e": (
            round(float(valid_fps.quantile(0.05)), 4)
            if len(valid_fps)
            else 0.0
        ),
        "mean_preprocess_ms": round(
            float(frames_df["preprocess_ms"].mean()), 4
        ),
        "mean_inference_ms": (
            round(float(valid_inf.mean()), 4)
            if len(valid_inf)
            else 0.0
        ),
        "median_inference_ms": (
            round(float(valid_inf.median()), 4)
            if len(valid_inf)
            else 0.0
        ),
        "p95_inference_ms": (
            round(float(valid_inf.quantile(0.95)), 4)
            if len(valid_inf)
            else 0.0
        ),
        "mean_postprocess_ms": round(
            float(frames_df["postprocess_ms"].mean()), 4
        ),
        "total_detections": int(total_detections),
        "fire_detections": int(fire_detections),
        "smoke_detections": int(smoke_detections),
        "frames_with_any_detection": int(
            any_detection_frames
        ),
        "frames_with_fire": int(fire_frames),
        "frames_with_smoke": int(smoke_frames),
        "any_detection_frame_rate_pct": round(
            100 * any_detection_frames / processed_frames, 3
        ),
        "fire_frame_rate_pct": round(
            100 * fire_frames / processed_frames, 3
        ),
        "smoke_frame_rate_pct": round(
            100 * smoke_frames / processed_frames, 3
        ),
        "mean_detection_confidence": (
            round(float(all_conf.mean()), 4)
            if len(all_conf)
            else np.nan
        ),
        "max_detection_confidence": (
            round(float(all_conf.max()), 4)
            if len(all_conf)
            else np.nan
        ),
    }

    pd.DataFrame([summary]).to_csv(
        run_dir / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with open(
        run_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    try:
        with pd.ExcelWriter(
            run_dir / "results.xlsx",
            engine="openpyxl",
        ) as writer_xlsx:
            pd.DataFrame([summary]).to_excel(
                writer_xlsx,
                sheet_name="Summary",
                index=False,
            )
            frames_df.to_excel(
                writer_xlsx,
                sheet_name="Frames",
                index=False,
            )
            det_df.to_excel(
                writer_xlsx,
                sheet_name="Detections",
                index=False,
            )
    except Exception as exc:
        print(f"[WARN] Excel export skipped: {exc}")

    save_plots(frames_df, det_df, run_dir)
    update_master_summary(
        summary, Path(args.results_root)
    )

    print("\n=== TEST SAVED ===")
    print(f"Run folder: {run_dir.resolve()}")
    print(
        "Average end-to-end FPS: "
        f"{summary['avg_fps_e2e']}"
    )
    print(
        "Mean inference latency: "
        f"{summary['mean_inference_ms']} ms"
    )
    print(
        f"Total detections: "
        f"{summary['total_detections']}"
    )
    print(
        "Frames with fire: "
        f"{summary['frames_with_fire']} "
        f"({summary['fire_frame_rate_pct']}%)"
    )
    print(
        "Frames with smoke: "
        f"{summary['frames_with_smoke']} "
        f"({summary['smoke_frame_rate_pct']}%)"
    )
    if not args.no_save_video:
        print(
            f"Annotated video: "
            f"{video_out_path.resolve()}"
        )
    print(
        "Master table: "
        f"{(Path(args.results_root) / 'all_runs_summary.csv').resolve()}"
    )


if __name__ == "__main__":
    main()
