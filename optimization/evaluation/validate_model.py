# -*- coding: utf-8 -*-
"""Validate .pt or .engine on the labelled fire/smoke validation set."""
import argparse, json
from pathlib import Path
from ultralytics import YOLO


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0')
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--output', default=None)
    args=ap.parse_args()
    mp=Path(args.model)
    model=YOLO(str(mp), task='detect') if mp.suffix.lower()=='.engine' else YOLO(str(mp))
    r=model.val(data=args.data, imgsz=args.imgsz, device=args.device, batch=args.batch, plots=True, verbose=True)
    out={
        'model': str(mp.resolve()),
        'precision': float(r.box.mp),
        'recall': float(r.box.mr),
        'map50': float(r.box.map50),
        'map50_95': float(r.box.map),
    }
    print('\n=== VALIDATION METRICS ===')
    for k,v in out.items(): print(k, ':', v)
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2), encoding='utf-8')
        print('Saved:', Path(args.output).resolve())

if __name__=='__main__': main()
