# -*- coding: utf-8 -*-
"""YOLO11-S optimized Jetson benchmark. Supports TensorRT .engine and PyTorch .pt."""
import argparse, json, platform, subprocess, sys, time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO

MODEL_LABEL = "YOLO11-S-OPT"
DEFAULT_MODEL = "models/yolo11s_best_fp16.engine"


def parse_source(v):
    v=str(v).strip()
    return int(v) if v.isdigit() else v


def group_name(name):
    n=str(name).lower()
    if "fire" in n or "flame" in n: return "fire"
    if "smoke" in n: return "smoke"
    return "other"


def jetson_model():
    try: return Path('/proc/device-tree/model').read_text(errors='ignore').strip('\x00\n ')
    except: return 'Unknown Jetson'


def gpu_name(device):
    if str(device).lower()=='cpu': return 'CPU'
    if torch.cuda.is_available():
        try: return torch.cuda.get_device_name(int(str(device).split(',')[0]))
        except: return 'CUDA device 0'
    return 'CUDA unavailable'


def open_camera(source, width, height, fps):
    if isinstance(source, int):
        cap=cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release(); cap=cv2.VideoCapture(source)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
            cap.set(cv2.CAP_PROP_FPS,fps)
        return cap
    return cv2.VideoCapture(source)


def writer_for(path, fps, size):
    w=cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if w.isOpened(): return w,path
    alt=path.with_suffix('.avi')
    w=cv2.VideoWriter(str(alt), cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    if not w.isOpened(): raise RuntimeError('Cannot create output video')
    return w,alt


def draw(img,text,xy,scale=.55,th=1):
    cv2.putText(img,str(text),xy,cv2.FONT_HERSHEY_SIMPLEX,scale,(235,235,235),th,cv2.LINE_AA)


def dashboard(frame,m,duration):
    h,w=frame.shape[:2]
    panel=np.full((h,360,3),24,np.uint8)
    y=34
    draw(panel,MODEL_LABEL,(20,y),.82,2); y+=34
    draw(panel,f"Device: {m['device']}",(20,y),.48); y+=25
    draw(panel,f"Test: {m['test']}",(20,y),.52); y+=40
    draw(panel,f"State: {m['state']}",(20,y),.60,2); y+=34
    draw(panel,f"Elapsed: {m['elapsed']:.1f} s",(20,y)); y+=27
    draw(panel,f"Current FPS: {m['fps']:.2f}",(20,y)); y+=27
    draw(panel,f"Average FPS: {m['avg_fps']:.2f}",(20,y)); y+=27
    draw(panel,f"Inference: {m['inf']:.2f} ms",(20,y)); y+=42
    draw(panel,f"Detections now: {m['det']}",(20,y)); y+=27
    draw(panel,f"Fire now: {m['fire']}",(20,y)); y+=27
    draw(panel,f"Smoke now: {m['smoke']}",(20,y)); y+=42
    draw(panel,'Controls',(20,y),.60,2); y+=29
    draw(panel,'SPACE/S  Start',(20,y),.50); y+=24
    draw(panel,'P        Pause/resume',(20,y),.50); y+=24
    draw(panel,'Q/ESC    Save & quit',(20,y),.50)
    if duration>0:
        p=min(max(m['elapsed']/duration,0),1)
        x0,y0,x1,y1=20,h-48,340,h-28
        cv2.rectangle(panel,(x0,y0),(x1,y1),(90,90,90),1)
        cv2.rectangle(panel,(x0+2,y0+2),(int(x0+(x1-x0)*p),y1-2),(200,200,200),-1)
    return np.hstack([frame,panel])


def save_plots(frames, dets, out):
    if frames.empty: return
    fig=plt.figure(figsize=(9,4.8)); plt.plot(frames.elapsed_s,frames.fps_e2e); plt.xlabel('Elapsed time (s)'); plt.ylabel('End-to-end FPS'); plt.title(f'{MODEL_LABEL} - FPS over time'); plt.grid(alpha=.25); plt.tight_layout(); fig.savefig(out/'01_fps_over_time.png',dpi=180); plt.close(fig)
    fig=plt.figure(figsize=(9,4.8)); plt.plot(frames.elapsed_s,frames.inference_ms); plt.xlabel('Elapsed time (s)'); plt.ylabel('Inference latency (ms)'); plt.title(f'{MODEL_LABEL} - Inference latency over time'); plt.grid(alpha=.25); plt.tight_layout(); fig.savefig(out/'02_inference_latency_over_time.png',dpi=180); plt.close(fig)
    vals=[int(frames.fire_count.sum()),int(frames.smoke_count.sum()),int(frames.other_count.sum())]
    fig=plt.figure(figsize=(7,4.8)); plt.bar(['Fire','Smoke','Other'],vals); plt.ylabel('Detection boxes across frames'); plt.title(f'{MODEL_LABEL} - Detection counts'); plt.tight_layout(); fig.savefig(out/'03_detection_counts.png',dpi=180); plt.close(fig)
    if not dets.empty:
        fig=plt.figure(figsize=(8,4.8)); plt.hist(dets.confidence.dropna(),bins=20); plt.xlabel('Confidence'); plt.ylabel('Detections'); plt.title(f'{MODEL_LABEL} - Confidence distribution'); plt.tight_layout(); fig.savefig(out/'04_confidence_distribution.png',dpi=180); plt.close(fig)


def update_master(summary, root):
    root.mkdir(parents=True,exist_ok=True)
    csvp=root/'all_runs_summary.csv'; xlsxp=root/'all_runs_summary.xlsx'
    new=pd.DataFrame([summary])
    if csvp.exists():
        try: new=pd.concat([pd.read_csv(csvp),new],ignore_index=True)
        except: pass
    if 'run_id' in new.columns: new=new.drop_duplicates('run_id',keep='last')
    new.to_csv(csvp,index=False,encoding='utf-8-sig')
    try: new.to_excel(xlsxp,index=False)
    except Exception as e: print('[WARN] master xlsx skipped:',e)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default=DEFAULT_MODEL)
    ap.add_argument('--source',default='0')
    ap.add_argument('--device',default='0')
    ap.add_argument('--conf',type=float,default=.15, help='Keep 0.15 for direct baseline comparison; change only after threshold calibration')
    ap.add_argument('--iou',type=float,default=.70)
    ap.add_argument('--imgsz',type=int,default=640)
    ap.add_argument('--duration',type=float,default=60)
    ap.add_argument('--test-id',default='test01')
    ap.add_argument('--platform-tag',default='JETSON')
    ap.add_argument('--results-root',default='results_optimized')
    ap.add_argument('--width',type=int,default=1280)
    ap.add_argument('--height',type=int,default=720)
    ap.add_argument('--camera-fps',type=float,default=30)
    ap.add_argument('--warmup',type=int,default=5)
    ap.add_argument('--no-save-video',action='store_true')
    ap.add_argument('--no-show',action='store_true')
    ap.add_argument('--auto-start',action='store_true')
    args=ap.parse_args()

    mp=Path(args.model)
    if not mp.exists():
        print('[ERROR] model not found:',mp.resolve()); sys.exit(2)
    if str(args.device).lower()!='cpu' and not torch.cuda.is_available():
        print('[ERROR] CUDA unavailable. Check Jetson PyTorch/CUDA first.'); sys.exit(3)

    source=parse_source(args.source)
    print('\n===',MODEL_LABEL,'JETSON BENCHMARK ===')
    print('Jetson:',jetson_model())
    print('Model:',mp.resolve())
    print('Device:',gpu_name(args.device))
    print('Torch:',torch.__version__,'CUDA:',torch.version.cuda)
    print('Ultralytics:',ultralytics.__version__,'OpenCV:',cv2.__version__)

    model=YOLO(str(mp), task='detect') if mp.suffix.lower()=='.engine' else YOLO(str(mp))
    cap=open_camera(source,args.width,args.height,args.camera_fps)
    if not cap.isOpened():
        print('[ERROR] cannot open source',source); print('Try: ls -l /dev/video*  and use --source 1 if needed'); sys.exit(4)

    frame=None; ok=False
    for _ in range(12):
        ok,frame=cap.read(); time.sleep(.01)
    if not ok or frame is None: raise RuntimeError('Camera opened but no frame could be read')
    ah,aw=frame.shape[:2]
    rfps=cap.get(cv2.CAP_PROP_FPS)
    if rfps<=1 or rfps>240: rfps=args.camera_fps
    print(f'Capture: {aw}x{ah} @ reported {rfps:.2f} FPS')

    for _ in range(max(0,args.warmup)):
        model.predict(frame,imgsz=args.imgsz,conf=args.conf,iou=args.iou,device=args.device,verbose=False)

    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id=f'{args.platform_tag}_{MODEL_LABEL}_{args.test_id}_{stamp}'
    out=Path(args.results_root)/args.platform_tag/MODEL_LABEL/args.test_id/stamp
    out.mkdir(parents=True,exist_ok=True)
    writer=None; video_path=out/'annotated.mp4'
    frame_rows=[]; det_rows=[]
    auto=args.auto_start or args.no_show
    started=auto; paused=False; start=time.perf_counter() if auto else None; last_end=None
    processed=tot=fire_det=smoke_det=fire_frames=smoke_frames=any_frames=0
    current=frame

    if started: print('FORMAL TEST AUTO-STARTED.')
    else:
        print('\nREADY: phone video at 00:00 -> press SPACE/S -> start phone video immediately.')

    while True:
        if current is None:
            ok,frame=cap.read()
            if not ok or frame is None: print('[WARN] camera read failed'); break
        else: frame=current; current=None

        if paused and started:
            if not args.no_show:
                elapsed=time.perf_counter()-start
                m={'device':gpu_name(args.device),'test':args.test_id,'state':'PAUSED','elapsed':elapsed,'fps':0,'avg_fps':0,'inf':0,'det':0,'fire':0,'smoke':0}
                cv2.imshow(f'{MODEL_LABEL} Fire/Smoke Benchmark - Jetson',dashboard(frame.copy(),m,args.duration))
                k=cv2.waitKey(30)&0xFF
                if k in (ord('p'),ord('P')): paused=False
                elif k in (ord('q'),ord('Q'),27): break
            else: time.sleep(.03)
            continue

        r=model.predict(frame,imgsz=args.imgsz,conf=args.conf,iou=args.iou,device=args.device,verbose=False)[0]
        annotated=r.plot(); sp=getattr(r,'speed',{}) or {}
        pre=float(sp.get('preprocess',0) or 0); inf=float(sp.get('inference',0) or 0); post=float(sp.get('postprocess',0) or 0)
        n=fn=sn=on=0; confs=[]; cc=Counter()
        b=getattr(r,'boxes',None)
        if b is not None and len(b)>0:
            xy=b.xyxy.detach().cpu().numpy(); cf=b.conf.detach().cpu().numpy(); cl=b.cls.detach().cpu().numpy().astype(int); names=r.names
            for box,c,cid in zip(xy,cf,cl):
                name=str(names.get(int(cid),cid) if isinstance(names,dict) else names[int(cid)])
                g=group_name(name); n+=1; confs.append(float(c)); cc[name]+=1
                if g=='fire': fn+=1
                elif g=='smoke': sn+=1
                else: on+=1
                if started:
                    det_rows.append({'run_id':run_id,'frame_idx':processed+1,'elapsed_s':time.perf_counter()-start,'class_id':int(cid),'class_name':name,'group':g,'confidence':float(c),'x1':float(box[0]),'y1':float(box[1]),'x2':float(box[2]),'y2':float(box[3])})

        now=time.perf_counter(); fps=0 if last_end is None else (1/(now-last_end) if now>last_end else 0); last_end=now
        elapsed=0 if start is None else now-start
        if started:
            processed+=1; tot+=n; fire_det+=fn; smoke_det+=sn
            if n>0: any_frames+=1
            if fn>0: fire_frames+=1
            if sn>0: smoke_frames+=1
            frame_rows.append({'run_id':run_id,'model':MODEL_LABEL,'platform_tag':args.platform_tag,'test_id':args.test_id,'frame_idx':processed,'elapsed_s':elapsed,'fps_e2e':fps,'preprocess_ms':pre,'inference_ms':inf,'postprocess_ms':post,'pipeline_model_ms':pre+inf+post,'detection_count':n,'fire_count':fn,'smoke_count':sn,'other_count':on,'mean_confidence':float(np.mean(confs)) if confs else np.nan,'max_confidence':float(np.max(confs)) if confs else np.nan,'class_counts_json':json.dumps(dict(cc),ensure_ascii=False)})
            if not args.no_save_video:
                if writer is None: writer,video_path=writer_for(video_path,rfps,(annotated.shape[1],annotated.shape[0]))
                writer.write(annotated)

        vals=[x['fps_e2e'] for x in frame_rows if x['fps_e2e']>0]
        avg=float(np.mean(vals)) if vals else 0
        if not args.no_show:
            state='RUNNING' if started else 'READY - press SPACE/S'
            m={'device':gpu_name(args.device),'test':args.test_id,'state':state,'elapsed':elapsed,'fps':fps,'avg_fps':avg,'inf':inf,'det':n,'fire':fn,'smoke':sn}
            cv2.imshow(f'{MODEL_LABEL} Fire/Smoke Benchmark - Jetson',dashboard(annotated,m,args.duration))

        if started and args.duration>0 and elapsed>=args.duration:
            print(f'Reached {args.duration:.1f} seconds. Finishing test...'); break

        if not args.no_show:
            k=cv2.waitKey(1)&0xFF
            if k in (ord('q'),ord('Q'),27): break
            if k in (ord(' '),ord('s'),ord('S')) and not started:
                started=True; start=time.perf_counter(); last_end=None; print('FORMAL TEST STARTED.')
            elif k in (ord('p'),ord('P')) and started: paused=True

    cap.release()
    if writer is not None: writer.release()
    if not args.no_show: cv2.destroyAllWindows()
    if not started or not frame_rows: print('No formal test data recorded.'); return

    fdf=pd.DataFrame(frame_rows); ddf=pd.DataFrame(det_rows)
    fdf.to_csv(out/'frames.csv',index=False,encoding='utf-8-sig'); ddf.to_csv(out/'detections.csv',index=False,encoding='utf-8-sig')
    vf=fdf.loc[fdf.fps_e2e>0,'fps_e2e']; vi=fdf.inference_ms.dropna(); ac=ddf.confidence.dropna() if not ddf.empty else pd.Series(dtype=float)
    summary={
        'run_id':run_id,'timestamp':stamp,'platform_tag':args.platform_tag,'model':MODEL_LABEL,'model_path':str(mp.resolve()),'model_size_mb':round(mp.stat().st_size/1024/1024,3),'test_id':args.test_id,'source':str(args.source),'device_arg':str(args.device),'device_name':gpu_name(args.device),'jetson_model':jetson_model(),'os':platform.platform(),'python_version':platform.python_version(),'torch_version':torch.__version__,'torch_cuda_version':str(torch.version.cuda),'ultralytics_version':ultralytics.__version__,'opencv_version':cv2.__version__,'imgsz':args.imgsz,'conf':args.conf,'iou':args.iou,'capture_width':aw,'capture_height':ah,'capture_reported_fps':rfps,'duration_s':round(float(fdf.elapsed_s.max()),3),'processed_frames':processed,'avg_fps_e2e':round(float(vf.mean()) if len(vf) else 0,4),'median_fps_e2e':round(float(vf.median()) if len(vf) else 0,4),'p05_fps_e2e':round(float(vf.quantile(.05)) if len(vf) else 0,4),'mean_preprocess_ms':round(float(fdf.preprocess_ms.mean()),4),'mean_inference_ms':round(float(vi.mean()) if len(vi) else 0,4),'median_inference_ms':round(float(vi.median()) if len(vi) else 0,4),'p95_inference_ms':round(float(vi.quantile(.95)) if len(vi) else 0,4),'mean_postprocess_ms':round(float(fdf.postprocess_ms.mean()),4),'total_detections':tot,'fire_detections':fire_det,'smoke_detections':smoke_det,'frames_with_any_detection':any_frames,'frames_with_fire':fire_frames,'frames_with_smoke':smoke_frames,'any_detection_frame_rate_pct':round(100*any_frames/processed,3),'fire_frame_rate_pct':round(100*fire_frames/processed,3),'smoke_frame_rate_pct':round(100*smoke_frames/processed,3),'mean_detection_confidence':round(float(ac.mean()),4) if len(ac) else np.nan,'max_detection_confidence':round(float(ac.max()),4) if len(ac) else np.nan
    }
    pd.DataFrame([summary]).to_csv(out/'summary.csv',index=False,encoding='utf-8-sig')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    try:
        with pd.ExcelWriter(out/'results.xlsx',engine='openpyxl') as x:
            pd.DataFrame([summary]).to_excel(x,'Summary',index=False); fdf.to_excel(x,'Frames',index=False); ddf.to_excel(x,'Detections',index=False)
    except Exception as e: print('[WARN] xlsx skipped:',e)
    save_plots(fdf,ddf,out); update_master(summary,Path(args.results_root))
    print('\n=== TEST SAVED ===')
    print('Run folder:',out.resolve())
    print('Average end-to-end FPS:',summary['avg_fps_e2e'])
    print('Mean inference latency:',summary['mean_inference_ms'],'ms')
    print('Total detections:',summary['total_detections'])
    print('Frames with fire:',summary['frames_with_fire'],f"({summary['fire_frame_rate_pct']}%)")
    print('Frames with smoke:',summary['frames_with_smoke'],f"({summary['smoke_frame_rate_pct']}%)")
    if not args.no_save_video: print('Annotated video:',video_path.resolve())
    print('Master table:',(Path(args.results_root)/'all_runs_summary.csv').resolve())

if __name__=='__main__': main()
