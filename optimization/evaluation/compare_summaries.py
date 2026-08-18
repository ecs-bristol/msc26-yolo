# -*- coding: utf-8 -*-
"""Compare one baseline summary.json with one optimized summary.json."""
import argparse, json
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def val(d, k, default=0.0):
    x = d.get(k, default)
    try: return float(x)
    except: return default


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--optimized', required=True)
    args=ap.parse_args()
    b=load(args.baseline); o=load(args.optimized)
    b_lat=val(b,'mean_inference_ms'); o_lat=val(o,'mean_inference_ms')
    b_fps=val(b,'avg_fps_e2e'); o_fps=val(o,'avg_fps_e2e')
    speedup=(b_lat/o_lat) if o_lat>0 else 0
    reduction=((b_lat-o_lat)/b_lat*100) if b_lat>0 else 0
    print('\n=== YOLO11 OPTIMIZATION COMPARISON ===')
    print(f"Baseline model:   {b.get('model_path','')}")
    print(f"Optimized model:  {o.get('model_path','')}")
    print(f"Mean inference:   {b_lat:.3f} -> {o_lat:.3f} ms")
    print(f"Latency reduction:{reduction:.2f}%")
    print(f"Inference speedup:{speedup:.2f}x")
    print(f"End-to-end FPS:   {b_fps:.3f} -> {o_fps:.3f}")
    print(f"Model size:       {val(b,'model_size_mb'):.3f} -> {val(o,'model_size_mb'):.3f} MB")
    print(f"Any-det frame %:  {val(b,'any_detection_frame_rate_pct'):.3f} -> {val(o,'any_detection_frame_rate_pct'):.3f}")
    print('\nAccuracy metrics and runtime performance are reported separately.')

if __name__=='__main__': main()
