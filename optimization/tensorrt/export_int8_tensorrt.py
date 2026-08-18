# -*- coding: utf-8 -*-
"""Build a Jetson-local TensorRT INT8 engine for YOLO11-S.
Requires a representative fire/smoke dataset YAML for INT8 calibration.
Run on the SAME Jetson used for deployment.
"""
import argparse
import platform
import shutil
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/yolo11s_best.pt')
    ap.add_argument('--data', required=True, help='Path to fire/smoke data.yaml for INT8 calibration')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8, help='Calibration max batch; reduce if memory is insufficient')
    ap.add_argument('--fraction', type=float, default=1.0)
    ap.add_argument('--device', default='0')
    ap.add_argument('--workspace', type=float, default=None)
    ap.add_argument('--output', default='models/yolo11s_best_int8.engine')
    args = ap.parse_args()

    src = Path(args.model)
    data = Path(args.data)
    if not src.exists():
        raise FileNotFoundError(f'Model not found: {src.resolve()}')
    if not data.exists():
        raise FileNotFoundError(f'data.yaml not found: {data.resolve()}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Run INT8 calibration/export on the target Jetson.')

    print('=== YOLO11-S -> TensorRT INT8 ===')
    print('Platform:', platform.platform())
    print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
    print('Ultralytics:', ultralytics.__version__)
    print('GPU:', torch.cuda.get_device_name(0))
    print('Calibration YAML:', data.resolve())

    model = YOLO(str(src))
    common = dict(
        format='engine',
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        dynamic=True,
        simplify=True,
        workspace=args.workspace,
        data=str(data),
        fraction=args.fraction,
    )
    try:
        exported = model.export(quantize=8, **common)
    except Exception as first_error:
        msg = str(first_error).lower()
        if 'quantize' not in msg and 'argument' not in msg and 'valid' not in msg:
            raise
        print('[INFO] quantize=8 was not accepted; retrying with legacy int8=True.')
        exported = model.export(int8=True, **common)

    exported_path = Path(str(exported))
    if not exported_path.exists():
        candidate = src.with_suffix('.engine')
        if candidate.exists():
            exported_path = candidate
        else:
            raise FileNotFoundError(f'Export finished but engine file was not found: {exported}')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != out.resolve():
        shutil.copy2(exported_path, out)

    print('\n=== INT8 EXPORT COMPLETE ===')
    print('Engine:', out.resolve())
    print('Size: %.2f MB' % (out.stat().st_size / 1024 / 1024))
    print('IMPORTANT: re-evaluate validation accuracy and re-select the confidence threshold for INT8.')


if __name__ == '__main__':
    main()
