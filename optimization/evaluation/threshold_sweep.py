# -*- coding: utf-8 -*-
"""Deployment-threshold sweep on a labelled validation set.
Reports precision/recall/F1 at fixed confidence thresholds. mAP is reported from standard validation and is not selected by the threshold sweep.
"""
import argparse, csv
from pathlib import Path
from ultralytics import YOLO


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--device', default='0')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--thresholds', default='0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50')
    ap.add_argument('--output', default='threshold_sweep.csv')
    args=ap.parse_args()
    mp=Path(args.model)
    model=YOLO(str(mp), task='detect') if mp.suffix.lower()=='.engine' else YOLO(str(mp))
    ths=[float(x.strip()) for x in args.thresholds.split(',') if x.strip()]
    rows=[]
    for conf in ths:
        print(f'\n--- conf={conf:.2f} ---')
        r=model.val(data=args.data, imgsz=args.imgsz, device=args.device, batch=1, conf=conf, plots=False, verbose=False)
        p=float(r.box.mp); rec=float(r.box.mr); f1=2*p*rec/(p+rec) if p+rec else 0.0
        rows.append({'conf':conf,'precision':p,'recall':rec,'f1':f1,'map50':float(r.box.map50),'map50_95':float(r.box.map)})
        print(f'P={p:.4f} R={rec:.4f} F1={f1:.4f}')
    with open(args.output,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    best=max(rows,key=lambda x:x['f1'])
    print('\n=== BEST FIXED THRESHOLD BY F1 ===')
    print(best)
    print('Saved:', Path(args.output).resolve())

if __name__=='__main__': main()
