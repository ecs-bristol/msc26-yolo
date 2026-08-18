# -*- coding: utf-8 -*-
"""Combine saved validation JSON and runtime summary JSON into one CSV row per variant."""
import argparse, csv, json
from pathlib import Path


def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variant', action='append', nargs=3, metavar=('NAME','VAL_JSON','RUN_JSON'), required=True,
                    help='Repeat: --variant FP32 val_fp32.json summary_fp32.json')
    ap.add_argument('--output', default='yolo11_optimization_comparison.csv')
    args=ap.parse_args()
    rows=[]
    for name,vp,rp in args.variant:
        v=load(vp); r=load(rp)
        rows.append({
            'variant':name,
            'precision':v.get('precision'), 'recall':v.get('recall'), 'map50':v.get('map50'), 'map50_95':v.get('map50_95'),
            'model_size_mb':r.get('model_size_mb'), 'mean_inference_ms':r.get('mean_inference_ms'),
            'p95_inference_ms':r.get('p95_inference_ms'), 'avg_fps_e2e':r.get('avg_fps_e2e'),
            'any_detection_frame_rate_pct':r.get('any_detection_frame_rate_pct')
        })
    with open(args.output,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print('Saved:', Path(args.output).resolve())

if __name__=='__main__': main()
