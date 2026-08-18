# -*- coding: utf-8 -*-
"""Build a Jetson-local TensorRT FP16 engine from the trained YOLO11-S model.
Run this export on the target Jetson deployment environment.
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
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--device', default='0')
    ap.add_argument('--workspace', type=float, default=None,
                    help='TensorRT workspace GiB. Default None lets TensorRT auto-allocate.')
    ap.add_argument('--output', default='models/yolo11s_best_fp16.engine')
    args = ap.parse_args()

    src = Path(args.model)
    if not src.exists():
        raise FileNotFoundError(f'Model not found: {src.resolve()}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Run this on the Jetson GPU environment.')

    print('=== YOLO11-S -> TensorRT FP16 ===')
    print('Platform:', platform.platform())
    print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
    print('Ultralytics:', ultralytics.__version__)
    print('GPU:', torch.cuda.get_device_name(0))
    print('Input model:', src.resolve())
    print('imgsz:', args.imgsz, 'batch:', args.batch)

    model = YOLO(str(src))
    common = dict(
        format='engine',
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        dynamic=False,
        simplify=True,
        workspace=args.workspace,
    )

    # Ultralytics >= current releases use quantize=16. Older 8.x builds may
    # still accept half=True. This fallback keeps the script usable on both.
    try:
        exported = model.export(quantize=16, **common)
    except Exception as first_error:
        msg = str(first_error).lower()
        if 'quantize' not in msg and 'argument' not in msg and 'valid' not in msg:
            raise
        print('[INFO] quantize=16 was not accepted; retrying with legacy half=True.')
        exported = model.export(half=True, **common)

    exported_path = Path(str(exported))
    if not exported_path.exists():
        # Typical output is next to .pt with .engine suffix.
        candidate = src.with_suffix('.engine')
        if candidate.exists():
            exported_path = candidate
        else:
            raise FileNotFoundError(f'Export finished but engine file was not found: {exported}')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != out.resolve():
        shutil.copy2(exported_path, out)

    print('\n=== FP16 EXPORT COMPLETE ===')
    print('Engine:', out.resolve())
    print('Size: %.2f MB' % (out.stat().st_size / 1024 / 1024))
    print('\nNext command:')
    print(f'python 03_run_yolo11_optimized.py --model "{out}" --source 0 --device 0 --conf 0.15 --imgsz {args.imgsz} --duration 60 --test-id fp16_camera')


if __name__ == '__main__':
    main()
