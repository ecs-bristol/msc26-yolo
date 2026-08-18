# -*- coding: utf-8 -*-
"""Jetson fire/smoke dynamic alarm Web system.

Features:
- Safe switching among YOLOv8-S / YOLOv9-S / YOLO11 / YOLO26-S (one loaded at a time)
- Shared MobileNetV3-Small CCTV scene classifier (default/fire/smoke)
- Scene assistance ON/OFF, interval 1/3/5
- Dynamic alarm: fire 3 frames, smoke 5 frames, clear 10 safe frames
- Live MJPEG camera, FPS/latency charts, alarm history, model comparison
- CSV/JSON/video/alarm screenshots for experiment analysis

Expected model classes:
- YOLO: 0=smoke, 1=fire (class names are preferred if present)
- Scene classifier: 0=default, 1=fire, 2=smoke

Run:
    python3 jetson_fire_alarm_final.py
Open:
    http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse, gc, json, os, platform, threading, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import ultralytics
from flask import Flask, Response, jsonify, request, render_template_string
from ultralytics import YOLO

try:
    import psutil
except Exception:
    psutil = None
try:
    from torchvision.models import mobilenet_v3_small
except Exception:
    mobilenet_v3_small = None

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_ROOT = ROOT / "results_alarm"
MASTER_CSV = RESULTS_ROOT / "master_benchmark.csv"

RISK = dict(
    fire_high=0.45, fire_low=0.30,
    smoke_high=0.45, smoke_low=0.35,
    scene_threshold=0.70,
    fire_alarm_frames=3, smoke_alarm_frames=5, clear_frames=10,
)
DEFAULT = dict(source="0", device="0", imgsz=640, conf=0.15, iou=0.70,
               width=1280, height=720, camera_fps=30.0)
SCENE_CLASSES = ("default", "fire", "smoke")

app = Flask(__name__)
lock = threading.RLock()
stop_event = threading.Event()
pause_event = threading.Event()
worker_thread = None
latest_jpeg = None


def jetson_model():
    try:
        return Path('/proc/device-tree/model').read_text(errors='ignore').strip('\x00\n ')
    except Exception:
        return 'Unknown Jetson'


def gpu_name(device='0'):
    if str(device).lower() == 'cpu': return 'CPU'
    if torch.cuda.is_available():
        try: return torch.cuda.get_device_name(int(str(device).split(',')[0]))
        except Exception: return 'CUDA device 0'
    return 'CUDA unavailable'


def rss_mb():
    if psutil is None: return 0.0
    try: return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception: return 0.0


def cuda_mem():
    if not torch.cuda.is_available(): return (0.0, 0.0)
    try:
        return (torch.cuda.memory_allocated()/1024/1024,
                torch.cuda.memory_reserved()/1024/1024)
    except Exception:
        return (0.0, 0.0)


def parse_source(v):
    s = str(v).strip(); return int(s) if s.isdigit() else s


def open_camera(source, w, h, fps):
    src = parse_source(source)
    if isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release(); cap = cv2.VideoCapture(src)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
            cap.set(cv2.CAP_PROP_FPS, float(fps))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return cv2.VideoCapture(src)


def encode_jpeg(img):
    ok, b = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return b.tobytes() if ok else None


def placeholder():
    x = np.full((720,1280,3),18,np.uint8)
    cv2.putText(x,'Waiting for camera...',(70,365),cv2.FONT_HERSHEY_SIMPLEX,1.0,(225,225,225),2,cv2.LINE_AA)
    return encode_jpeg(x)


def model_label(p: Path):
    n = p.stem.lower().replace('_','').replace('-','')
    if 'yolov8' in n or 'yolo8' in n: return 'YOLOv8-S'
    if 'yolov9' in n or 'yolo9' in n: return 'YOLOv9-S'
    if 'yolo11' in n: return 'YOLO11'
    if 'yolo26' in n: return 'YOLO26-S'
    return p.stem if 'yolo' in n else None


def is_scene(p: Path):
    n=p.stem.lower()
    return any(k in n for k in ('scene','mobilenet','cctv','classifier','classification'))


def discover_models():
    det, scene = [], []
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.rglob('*')):
            if not p.is_file() or p.suffix.lower() not in {'.pt','.pth','.onnx','.engine','.ts','.torchscript'}: continue
            rel=str(p.relative_to(ROOT))
            item=dict(key=rel,path=rel,size_mb=round(p.stat().st_size/1024/1024,2))
            if is_scene(p):
                item['label']=p.stem; scene.append(item)
            else:
                lab=model_label(p)
                if lab: item['label']=lab; det.append(item)
    order={'YOLOv8-S':0,'YOLOv9-S':1,'YOLO11':2,'YOLO26-S':3}
    det.sort(key=lambda x:(order.get(x['label'],99),x['path']))
    return det, scene


class SceneClassifier:
    """MobileNetV3 scene classifier supporting full model/state_dict/TorchScript/ONNX."""
    def __init__(self, path, device='0', input_size=224):
        self.path=Path(path)
        if not self.path.is_absolute(): self.path=ROOT/self.path
        if not self.path.exists(): raise FileNotFoundError(f'Scene classifier not found: {self.path}')
        self.input_size=int(input_size)
        self.device=torch.device('cpu' if str(device).lower()=='cpu' or not torch.cuda.is_available()
                                 else f"cuda:{int(str(device).split(',')[0])}")
        self.model=None; self.net=None
        suf=self.path.suffix.lower()
        if suf=='.onnx':
            self.backend='onnx'; self.net=cv2.dnn.readNetFromONNX(str(self.path))
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            except Exception: pass
            return
        if suf in ('.ts','.torchscript'):
            self.backend='torchscript'; self.model=torch.jit.load(str(self.path),map_location=self.device).eval(); return
        obj=torch.load(str(self.path),map_location='cpu',weights_only=False)
        self.backend='pytorch'
        if isinstance(obj,torch.nn.Module):
            self.model=obj.to(self.device).eval(); return
        if mobilenet_v3_small is None: raise RuntimeError('torchvision is required for MobileNetV3 state_dict loading.')
        self.model=mobilenet_v3_small(weights=None)
        n=self.model.classifier[-1].in_features
        self.model.classifier[-1]=torch.nn.Linear(n,3)
        sd=obj
        if isinstance(obj,dict):
            if isinstance(obj.get('model_state_dict'),dict): sd=obj['model_state_dict']
            elif isinstance(obj.get('state_dict'),dict): sd=obj['state_dict']
            elif isinstance(obj.get('model'),dict): sd=obj['model']
        if not isinstance(sd,dict): raise RuntimeError('Unsupported scene checkpoint format.')
        sd={str(k).replace('module.','',1):v for k,v in sd.items()}
        self.model.load_state_dict(sd,strict=True); self.model.to(self.device).eval()

    def _prep(self,bgr):
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        rgb=cv2.resize(rgb,(self.input_size,self.input_size))
        x=rgb.astype(np.float32)/255.0
        x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        return np.transpose(x,(2,0,1))[None,...]

    @torch.inference_mode()
    def infer(self,frame):
        x=self._prep(frame)
        if self.backend=='onnx':
            self.net.setInput(x); z=self.net.forward().reshape(-1); z-=z.max(); e=np.exp(z); p=e/max(e.sum(),1e-12)
        else:
            z=self.model(torch.from_numpy(x).to(self.device))
            if isinstance(z,(tuple,list)): z=z[0]
            p=F.softmax(z,dim=1)[0].detach().cpu().numpy()
        return {c:float(p[i]) for i,c in enumerate(SCENE_CLASSES)}


class DynamicAlarm:
    def __init__(self):
        self.state='normal'; self.fire_streak=0; self.smoke_streak=0; self.safe_streak=0
        self.fire_start=None; self.smoke_start=None

    def update(self,idx,fc,sc,scene,scene_enabled=True):
        sf=float(scene.get('fire',0)) if scene_enabled else 0.0
        ss=float(scene.get('smoke',0)) if scene_enabled else 0.0
        fs=scene_enabled and sf>=RISK['scene_threshold']; ssup=scene_enabled and ss>=RISK['scene_threshold']
        fw=fc>=RISK['fire_low'] or fs; sw=sc>=RISK['smoke_low'] or ssup
        ft=fc>=RISK['fire_high'] or (fc>=RISK['fire_low'] and fs)
        st=sc>=RISK['smoke_high'] or (sc>=RISK['smoke_low'] and ssup)
        if ft:
            self.fire_streak+=1
            if self.fire_start is None:self.fire_start=idx
        else:self.fire_streak=0;self.fire_start=None
        if st:
            self.smoke_streak+=1
            if self.smoke_start is None:self.smoke_start=idx
        else:self.smoke_streak=0;self.smoke_start=None
        self.safe_streak=0 if (fw or sw) else self.safe_streak+1
        old=self.state; response=None
        if old in ('fire_alarm','smoke_alarm') and self.safe_streak<RISK['clear_frames']:
            if self.fire_streak>=RISK['fire_alarm_frames']: self.state='fire_alarm'; reason='fire_consecutive_frames'
            else:self.state=old;reason='alarm_hold_until_safe'
        elif self.safe_streak>=RISK['clear_frames']: self.state='normal';reason='safe_10_frames'
        elif self.fire_streak>=RISK['fire_alarm_frames']:
            self.state='fire_alarm';reason='fire_consecutive_frames';response=idx-self.fire_start+1
        elif self.smoke_streak>=RISK['smoke_alarm_frames']:
            self.state='smoke_alarm';reason='smoke_consecutive_frames';response=idx-self.smoke_start+1
        elif fw:self.state='fire_warning';reason='fire_low_or_scene_support'
        elif sw:self.state='smoke_warning';reason='smoke_low_or_scene_support'
        else:self.state='normal';reason='no_risk'
        return dict(state=self.state,changed=self.state!=old,reason=reason,response_frames=response,
                    fire_streak=self.fire_streak,smoke_streak=self.smoke_streak,safe_streak=self.safe_streak)


class RunSession:
    def __init__(self,cfg):
        self.cfg=cfg; self.label=cfg['model_label']
        self.det_path=ROOT/cfg['model_path']; self.scene_path=ROOT/cfg['scene_model_path'] if cfg['scene_model_path'] else None
        if not self.det_path.exists(): raise FileNotFoundError(self.det_path)
        if cfg['device']!='cpu' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        self.det=YOLO(str(self.det_path)); self.scene=None
        if cfg['scene_enabled']:
            if not self.scene_path: raise RuntimeError('Scene Assistance ON but no scene model selected.')
            self.scene=SceneClassifier(str(self.scene_path),cfg['device'])
        self.alarm=DynamicAlarm(); self.scene_probs={'default':1.0,'fire':0.0,'smoke':0.0}; self.last_scene_ms=0.0
        self.idx=0; self.start=time.perf_counter(); self.last_end=None; self.fps_vals=[]
        self.frames=[];self.events=[];self.event_history=[];self.total_yolo=0.0;self.total_pipe=0.0;self.total_scene=0.0;self.scene_runs=0;self.responses=[]
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S'); safe=''.join(c if c.isalnum() or c in '-_' else '_' for c in self.label)
        self.run_id=f'JETSON_{safe}_{cfg["test_id"]}_{stamp}'; self.run_dir=RESULTS_ROOT/self.run_id; self.shot_dir=self.run_dir/'alarm_shots'
        self.shot_dir.mkdir(parents=True,exist_ok=True)
        self.info=self._model_info()

    def _model_info(self):
        d={'model_size_mb':round(self.det_path.stat().st_size/1024/1024,3),'parameters':None,'gflops':None}
        try:
            r=self.det.info(verbose=False)
            if isinstance(r,(tuple,list)) and len(r)>=4:d['parameters']=int(r[1]);d['gflops']=float(r[3])
        except Exception:pass
        if d['parameters'] is None:
            try:d['parameters']=int(sum(p.numel() for p in self.det.model.parameters()))
            except Exception:pass
        return d

    def warmup(self):
        x=np.zeros((self.cfg['imgsz'],self.cfg['imgsz'],3),np.uint8)
        for _ in range(3):self.det.predict(x,imgsz=self.cfg['imgsz'],conf=self.cfg['conf'],iou=self.cfg['iou'],device=self.cfg['device'],verbose=False)
        if self.scene:self.scene.infer(x)

    @staticmethod
    def group(cid,name):
        n=str(name).lower()
        if 'fire' in n or 'flame' in n:return 'fire'
        if 'smoke' in n:return 'smoke'
        if int(cid)==0:return 'smoke'
        if int(cid)==1:return 'fire'
        return 'other'

    def process(self,frame):
        t0=time.perf_counter();self.idx+=1
        r=self.det.predict(frame,imgsz=self.cfg['imgsz'],conf=self.cfg['conf'],iou=self.cfg['iou'],device=self.cfg['device'],verbose=False)[0]
        sp=getattr(r,'speed',{}) or {}; yms=float(sp.get('inference',0) or 0)
        fc=sc=0.0;b=getattr(r,'boxes',None)
        if b is not None and len(b)>0:
            cf=b.conf.detach().cpu().numpy();cl=b.cls.detach().cpu().numpy().astype(int);names=r.names
            for c,cid in zip(cf,cl):
                name=str(names.get(int(cid),cid) if isinstance(names,dict) else names[int(cid)])
                g=self.group(cid,name)
                if g=='fire':fc=max(fc,float(c))
                elif g=='smoke':sc=max(sc,float(c))
        scene_updated=False; scene_current=0.0
        if self.scene:
            n=max(1,int(self.cfg['scene_interval']))
            if self.idx==1 or (self.idx-1)%n==0:
                s0=time.perf_counter();self.scene_probs=self.scene.infer(frame);self.last_scene_ms=(time.perf_counter()-s0)*1000
                scene_current=self.last_scene_ms;self.total_scene+=scene_current;self.scene_runs+=1;scene_updated=True
        dec=self.alarm.update(self.idx,fc,sc,self.scene_probs,self.cfg['scene_enabled'])
        ann=r.plot();pipe=(time.perf_counter()-t0)*1000;now=time.perf_counter();fps=0 if self.last_end is None else 1/max(now-self.last_end,1e-9);self.last_end=now
        if fps>0:self.fps_vals.append(fps)
        avg=float(np.mean(self.fps_vals)) if self.fps_vals else 0.0;elapsed=now-self.start;ca,cr=cuda_mem()
        if dec['changed']:
            shot=''
            if dec['state'] in ('fire_alarm','smoke_alarm'):
                p=self.shot_dir/f"{dec['state']}_{self.idx:06d}_{datetime.now().strftime('%H%M%S_%f')}.jpg";cv2.imwrite(str(p),ann);shot=str(p)
            ev=dict(timestamp=datetime.now().isoformat(timespec='milliseconds'),frame_idx=self.idx,elapsed_s=elapsed,state=dec['state'],reason=dec['reason'],
                    max_fire_conf=fc,max_smoke_conf=sc,scene_default=self.scene_probs.get('default',0),scene_fire=self.scene_probs.get('fire',0),scene_smoke=self.scene_probs.get('smoke',0),
                    fire_streak=dec['fire_streak'],smoke_streak=dec['smoke_streak'],safe_streak=dec['safe_streak'],response_frames=dec['response_frames'],screenshot=shot)
            self.events.append(ev);pd.DataFrame(self.events).to_csv(self.run_dir/'alarm_events.csv',index=False,encoding='utf-8-sig')
            self.event_history.append(dict(state=dec['state'],reason=dec['reason'],frame_idx=self.idx,elapsed_s=elapsed));self.event_history=self.event_history[-20:]
            if dec['response_frames'] and avg>0:self.responses.append(dec['response_frames']/avg)
        self.total_yolo+=yms;self.total_pipe+=pipe
        row=dict(run_id=self.run_id,model=self.label,frame_idx=self.idx,elapsed_s=elapsed,fps=fps,avg_fps=avg,yolo_inference_ms=yms,
                 scene_inference_ms=scene_current,pipeline_ms=pipe,max_fire_conf=fc,max_smoke_conf=sc,scene_default=self.scene_probs.get('default',0),
                 scene_fire=self.scene_probs.get('fire',0),scene_smoke=self.scene_probs.get('smoke',0),scene_updated=scene_updated,alarm_state=dec['state'],alarm_reason=dec['reason'],
                 fire_streak=dec['fire_streak'],smoke_streak=dec['smoke_streak'],safe_streak=dec['safe_streak'],rss_mb=rss_mb(),cuda_allocated_mb=ca,cuda_reserved_mb=cr)
        self.frames.append(row);snap=dict(row);snap.update(scene_probs=dict(self.scene_probs),event_history=list(self.event_history),model_info=self.info,device_name=gpu_name(self.cfg['device']))
        return ann,snap

    def finalize(self):
        fdf=pd.DataFrame(self.frames)
        if not fdf.empty:fdf.to_csv(self.run_dir/'frames.csv',index=False,encoding='utf-8-sig')
        if self.events:pd.DataFrame(self.events).to_csv(self.run_dir/'alarm_events.csv',index=False,encoding='utf-8-sig')
        n=max(1,self.idx)
        s=dict(run_id=self.run_id,timestamp=datetime.now().isoformat(timespec='seconds'),model=self.label,model_path=str(self.det_path),scene_enabled=self.cfg['scene_enabled'],
               scene_model_path=str(self.scene_path) if self.scene_path else '',scene_interval=self.cfg['scene_interval'],frames=self.idx,duration_s=float(fdf.elapsed_s.max()) if not fdf.empty else 0.0,
               avg_fps=float(np.mean(self.fps_vals)) if self.fps_vals else 0.0,mean_yolo_inference_ms=self.total_yolo/n,mean_pipeline_ms=self.total_pipe/n,
               scene_runs=self.scene_runs,mean_scene_inference_ms_when_run=self.total_scene/max(1,self.scene_runs),scene_cost_ms_per_video_frame=self.total_scene/n,
               mean_rss_mb=float(fdf.rss_mb.mean()) if not fdf.empty else 0.0,peak_rss_mb=float(fdf.rss_mb.max()) if not fdf.empty else 0.0,
               peak_cuda_allocated_mb=float(fdf.cuda_allocated_mb.max()) if not fdf.empty else 0.0,mean_alarm_response_s=float(np.mean(self.responses)) if self.responses else None,
               final_alarm_state=self.alarm.state,alarm_state_changes=len(self.events),model_size_mb=self.info['model_size_mb'],parameters=self.info['parameters'],gflops=self.info['gflops'],
               jetson_model=jetson_model(),gpu=gpu_name(self.cfg['device']),python=platform.python_version(),torch=torch.__version__,torch_cuda=str(torch.version.cuda),ultralytics=ultralytics.__version__,opencv=cv2.__version__)
        (self.run_dir/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
        RESULTS_ROOT.mkdir(parents=True,exist_ok=True);new=pd.DataFrame([s])
        if MASTER_CSV.exists():
            try:new=pd.concat([pd.read_csv(MASTER_CSV),new],ignore_index=True)
            except Exception:pass
        new.drop_duplicates('run_id',keep='last').to_csv(MASTER_CSV,index=False,encoding='utf-8-sig')
        return s

    def close(self):
        self.scene=None;self.det=None;gc.collect()
        if torch.cuda.is_available():
            try:torch.cuda.empty_cache()
            except Exception:pass


def fresh_state():
    return dict(running=False,paused=False,status='Ready',error='',current_model_key='',current_model_path='',scene_model_path='',scene_enabled=True,scene_interval=3,run_dir='',snapshot=None,
                series=dict(time=[],fps=[],yolo_ms=[],pipeline_ms=[],rss_mb=[]),
                system=dict(jetson_model=jetson_model(),gpu=gpu_name(),torch=torch.__version__,torch_cuda=str(torch.version.cuda),cuda_available=bool(torch.cuda.is_available()),ultralytics=ultralytics.__version__,opencv=cv2.__version__,python=platform.python_version()))
STATE=fresh_state()


def worker(cfg):
    global latest_jpeg
    cap=None;writer=None;sess=None
    try:
        with lock:STATE.update(status='Loading models',error='',current_model_key=cfg['model_key'],current_model_path=cfg['model_path'],scene_model_path=cfg['scene_model_path'],scene_enabled=cfg['scene_enabled'],scene_interval=cfg['scene_interval'])
        sess=RunSession(cfg)
        with lock:STATE['status']='Warming up'
        sess.warmup();cap=open_camera(cfg['source'],cfg['width'],cfg['height'],cfg['camera_fps'])
        if not cap.isOpened():raise RuntimeError(f"Cannot open camera source {cfg['source']}")
        ok,frame=cap.read()
        if not ok or frame is None:raise RuntimeError('Camera opened but no frame could be read')
        h,w=frame.shape[:2];rfps=cap.get(cv2.CAP_PROP_FPS)
        if rfps<=1 or rfps>240:rfps=cfg['camera_fps']
        if cfg['save_video']:writer=cv2.VideoWriter(str(sess.run_dir/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),float(rfps),(w,h))
        with lock:STATE.update(running=True,status='Running',run_dir=str(sess.run_dir.resolve()))
        while not stop_event.is_set():
            if pause_event.is_set():
                with lock:STATE.update(paused=True,status='Paused')
                time.sleep(.05);continue
            with lock:STATE.update(paused=False,status='Running')
            ok,frame=cap.read()
            if not ok or frame is None:raise RuntimeError('Camera frame read failed')
            ann,snap=sess.process(frame);latest_jpeg=encode_jpeg(ann)
            if writer is not None and writer.isOpened():writer.write(ann)
            with lock:
                STATE['snapshot']=snap
                ser=STATE['series']
                for k,v in [('time',snap['elapsed_s']),('fps',snap['fps']),('yolo_ms',snap['yolo_inference_ms']),('pipeline_ms',snap['pipeline_ms']),('rss_mb',snap['rss_mb'])]:
                    ser[k].append(round(float(v),3));ser[k]=ser[k][-120:]
            if cfg['duration']>0 and snap['elapsed_s']>=cfg['duration']:break
    except Exception as e:
        print('[ERROR]',repr(e))
        with lock:STATE.update(error=str(e),status='Error')
    finally:
        if cap is not None:cap.release()
        if writer is not None:writer.release()
        if sess is not None:
            try:
                s=sess.finalize();print('\n=== RUN SAVED ===');print('Folder:',sess.run_dir.resolve());print('Average FPS:',round(s['avg_fps'],3));print('YOLO ms:',round(s['mean_yolo_inference_ms'],3))
            except Exception as e:print('[WARN] finalize:',e)
            sess.close()
        with lock:
            STATE['running']=False;STATE['paused']=False
            if not STATE['error']:STATE['status']='Finished'
        stop_event.clear();pause_event.clear()


def history_payload():
    if not MASTER_CSV.exists():return {'rows':[],'latest_by_model':[]}
    try:df=pd.read_csv(MASTER_CSV)
    except Exception:return {'rows':[],'latest_by_model':[]}
    if df.empty:return {'rows':[],'latest_by_model':[]}
    rows=[]
    for _,r in df.tail(100).iloc[::-1].iterrows():
        def val(c):
            x=r.get(c,None);return None if pd.isna(x) else (x.item() if isinstance(x,np.generic) else x)
        rows.append({c:val(c) for c in ['timestamp','model','scene_enabled','scene_interval','avg_fps','mean_yolo_inference_ms','mean_pipeline_ms','mean_rss_mb','peak_cuda_allocated_mb','mean_alarm_response_s','model_size_mb','parameters','gflops','run_id']})
    latest=df.sort_values('timestamp').drop_duplicates('model',keep='last'); comp=[]
    for _,r in latest.iterrows():
        comp.append(dict(model=str(r.get('model','')),avg_fps=float(r.get('avg_fps',0) or 0),mean_yolo_inference_ms=float(r.get('mean_yolo_inference_ms',0) or 0),mean_pipeline_ms=float(r.get('mean_pipeline_ms',0) or 0),mean_rss_mb=float(r.get('mean_rss_mb',0) or 0)))
    return {'rows':rows,'latest_by_model':comp}


HTML = r"""
<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Jetson Fire & Smoke Alarm</title>
<style>
:root{--bg:#0b1016;--card:#111923;--line:#263443;--text:#edf3f8;--muted:#91a1b0;--accent:#76b900}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}.shell{max-width:1580px;margin:auto;padding:18px}.top{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:15px}h1{font-size:21px;margin:0}.sub{font-size:12px;color:var(--muted);margin-top:4px}.state{padding:10px 15px;border:1px solid var(--line);border-radius:999px;font-weight:800}.grid{display:grid;grid-template-columns:minmax(0,1.62fr) minmax(345px,.73fr);gap:15px}.card{background:var(--card);border:1px solid var(--line);border-radius:15px;overflow:hidden}.head{padding:13px 15px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:12px;font-size:13px;font-weight:750}.tag{color:var(--accent);font-size:10px}.video{background:#05080b;min-height:430px;display:grid;place-items:center}.video img{display:block;width:100%;max-height:69vh;object-fit:contain}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line)}.metric{background:var(--card);padding:13px}.label{font-size:10px;color:var(--muted)}.value{font-size:22px;font-weight:800;margin-top:5px}.unit{font-size:11px;color:var(--muted)}.side{display:flex;flex-direction:column;gap:15px}.section{padding:14px}.formgrid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.full{grid-column:1/-1}label{display:block;font-size:10px;color:var(--muted);margin-bottom:5px}select,input{width:100%;padding:9px 10px;border-radius:9px;border:1px solid var(--line);background:#0c131b;color:var(--text);font-size:12px}.check{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:12px}.check input{width:auto}.buttons{display:grid;grid-template-columns:1.25fr 1fr 1fr;gap:8px;margin-top:12px}button{border:0;border-radius:9px;padding:10px;font-weight:750;cursor:pointer}.start{background:var(--accent);color:#091100}.pause{background:#273747;color:var(--text)}.stop{background:#42262b;color:#ffb3b6}button:disabled{opacity:.45}.tri{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.mini{border:1px solid var(--line);border-radius:11px;padding:10px}.mini b{font-size:19px;display:block}.mini span{font-size:9px;color:var(--muted)}.rows{display:grid;grid-template-columns:1fr auto;gap:7px 12px;font-size:11px;margin-top:12px}.k{color:var(--muted)}.v{text-align:right;max-width:210px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.charts{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-top:15px}.canvaswrap{height:225px;padding:10px 12px}canvas{width:100%;height:100%}.history{margin-top:15px}.tablewrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:11px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:var(--muted)}.bars{padding:13px}.barrow{display:grid;grid-template-columns:90px 1fr 72px;gap:9px;align-items:center;margin:10px 0;font-size:11px}.track{height:9px;border-radius:999px;background:#0a1118;overflow:hidden}.fill{height:100%;background:var(--accent)}.error{color:#ff9fa4;font-size:10px;margin-top:10px}.note{font-size:10px;color:var(--muted);margin-top:10px;line-height:1.5}@media(max-width:1050px){.grid{grid-template-columns:1fr}}@media(max-width:760px){.shell{padding:9px}.metrics{grid-template-columns:1fr 1fr}.charts{grid-template-columns:1fr}.formgrid{grid-template-columns:1fr}.full{grid-column:auto}}
</style></head><body><div class="shell"><div class="top"><div><h1>Edge Fire & Smoke Dynamic Alarm</h1><div class="sub">YOLO model switching + shared CCTV scene classifier + temporal alarm + Jetson benchmark</div></div><div id="alarm" class="state">NORMAL</div></div>
<div class="grid"><section class="card"><div class="head"><span>Live Camera</span><span id="pipeTag" class="tag">WAITING</span></div><div class="video"><img src="/video_feed" alt="live stream"></div><div class="metrics"><div class="metric"><div class="label">Current FPS</div><div id="fps" class="value">0.00</div></div><div class="metric"><div class="label">Average FPS</div><div id="avg" class="value">0.00</div></div><div class="metric"><div class="label">YOLO inference</div><div class="value"><span id="yms">0.00</span><span class="unit"> ms</span></div></div><div class="metric"><div class="label">Pipeline latency</div><div class="value"><span id="pms">0.00</span><span class="unit"> ms</span></div></div></div></section>
<aside class="side"><section class="card"><div class="head"><span>Pipeline Control</span><span class="tag">SAFE SWITCH</span></div><div class="section"><div class="formgrid"><div class="full"><label>Detection model</label><select id="model"></select></div><div class="full"><label>Shared scene classifier</label><select id="scene"></select></div><div><label>Scene interval</label><select id="interval"><option value="1">Every 1 frame</option><option value="3" selected>Every 3 frames</option><option value="5">Every 5 frames</option></select></div><div><label>Camera source</label><input id="source" value="0"></div><div><label>Benchmark duration (s)</label><input id="duration" type="number" min="0" value="60"></div><div><label>Test ID</label><input id="testid" value="test01"></div><div class="full check"><input id="sceneOn" type="checkbox" checked><label style="margin:0">Scene Assistance ON</label></div><div class="full check"><input id="videoOn" type="checkbox" checked><label style="margin:0">Save annotated video</label></div></div><div class="buttons"><button id="start" class="start">Start / Load</button><button id="pause" class="pause">Pause</button><button id="stop" class="stop">Stop & Save</button></div><div class="note">切换 YOLO 前先 Stop。系统一次只加载一个 YOLO + 一个 MobileNetV3，避免 Jetson GPU 内存被四个模型同时占用。</div><div id="error" class="error"></div></div></section>
<section class="card"><div class="head"><span>Fusion & Alarm</span><span class="tag">CURRENT FRAME</span></div><div class="section"><div class="tri"><div class="mini"><b id="fc">0.000</b><span>YOLO fire max</span></div><div class="mini"><b id="sc">0.000</b><span>YOLO smoke max</span></div><div class="mini"><b id="sceneCls">default</b><span>CCTV scene</span></div></div><div class="rows"><div class="k">Scene default</div><div id="sd" class="v">0.000</div><div class="k">Scene fire</div><div id="sf" class="v">0.000</div><div class="k">Scene smoke</div><div id="ss" class="v">0.000</div><div class="k">Classifier</div><div id="sceneMode" class="v">—</div><div class="k">Fire streak</div><div id="fst" class="v">0 / 3</div><div class="k">Smoke streak</div><div id="sst" class="v">0 / 5</div><div class="k">Safe streak</div><div id="safe" class="v">0 / 10</div><div class="k">Reason</div><div id="reason" class="v">—</div></div></div></section>
<section class="card"><div class="head"><span>Jetson Runtime</span><span class="tag">DEPLOYMENT</span></div><div class="section rows" style="margin-top:0"><div class="k">GPU</div><div id="gpu" class="v">—</div><div class="k">RSS memory</div><div id="rss" class="v">0 MB</div><div class="k">CUDA allocated</div><div id="cmem" class="v">0 MB</div><div class="k">Model size</div><div id="msize" class="v">—</div><div class="k">Parameters</div><div id="params" class="v">—</div><div class="k">GFLOPs</div><div id="gflops" class="v">—</div><div class="k">Run folder</div><div id="run" class="v">—</div></div></section></aside></div>
<div class="charts"><section class="card"><div class="head"><span>Real-time FPS</span><span class="tag">ROLLING</span></div><div class="canvaswrap"><canvas id="fpsChart"></canvas></div></section><section class="card"><div class="head"><span>Real-time Latency</span><span class="tag">YOLO / PIPELINE</span></div><div class="canvaswrap"><canvas id="latChart"></canvas></div></section></div>
<section class="card history"><div class="head"><span>Latest Model Comparison</span><span class="tag">COMPLETED RUNS</span></div><div id="bars" class="bars"></div><div class="tablewrap"><table><thead><tr><th>Model</th><th>Avg FPS</th><th>YOLO ms</th><th>Pipeline ms</th><th>RSS MB</th><th>Alarm response s</th><th>Scene</th><th>Interval</th></tr></thead><tbody id="tbody"></tbody></table></div></section></div>
<script>
const $=id=>document.getElementById(id),fmt=(x,n=2)=>Number(x||0).toFixed(n);
function alarmColor(s){return {normal:['#76b900','#081005'],smoke_warning:['#aab8c2','#091016'],smoke_alarm:['#e49a3a','#180e02'],fire_warning:['#ec7844','#190802'],fire_alarm:['#e45858','#1a0303']}[s]||['#536170','#fff']}
async function models(){const d=await(await fetch('/api/models')).json();$('model').innerHTML='';d.detectors.forEach(m=>{let o=document.createElement('option');o.value=m.key;o.textContent=`${m.label} · ${m.path} · ${m.size_mb} MB`;if(m.label==='YOLO11')o.selected=true;$('model').appendChild(o)});$('scene').innerHTML='';if(!d.scenes.length){let o=document.createElement('option');o.value='';o.textContent='No scene model found';$('scene').appendChild(o)}else d.scenes.forEach(m=>{let o=document.createElement('option');o.value=m.key;o.textContent=`${m.label} · ${m.path}`;$('scene').appendChild(o)})}
function canvas(id){let c=$(id),dpr=devicePixelRatio||1,r=c.getBoundingClientRect(),w=Math.max(1,Math.floor(r.width*dpr)),h=Math.max(1,Math.floor(r.height*dpr));if(c.width!==w||c.height!==h){c.width=w;c.height=h}let x=c.getContext('2d');x.setTransform(dpr,0,0,dpr,0,0);return [x,r.width,r.height]}
function chart(id,sets){let [x,W,H]=canvas(id);x.clearRect(0,0,W,H);let p={l:42,r:12,t:12,b:26},cw=W-p.l-p.r,ch=H-p.t-p.b,vals=sets.flatMap(s=>s.v).map(Number).filter(Number.isFinite),mx=Math.max(1,...vals)*1.08;x.strokeStyle='#293746';x.lineWidth=1;for(let i=0;i<=4;i++){let y=p.t+ch*i/4;x.beginPath();x.moveTo(p.l,y);x.lineTo(W-p.r,y);x.stroke()}x.fillStyle='#91a1b0';x.font='10px system-ui';for(let i=0;i<=4;i++)x.fillText((mx*(1-i/4)).toFixed(1),4,p.t+ch*i/4+3);let colors=['#76b900','#4fa6e8','#e49a3a'];sets.forEach((s,j)=>{if(!s.v.length)return;x.strokeStyle=colors[j];x.lineWidth=1.8;x.beginPath();s.v.forEach((v,i)=>{let px=p.l+(s.v.length<=1?0:cw*i/(s.v.length-1)),py=p.t+ch-Number(v)/mx*ch;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();x.fillStyle=colors[j];x.fillText(s.n,p.l+8+j*92,H-6)})}
async function refresh(){try{const d=await(await fetch('/api/status',{cache:'no-store'})).json(),s=d.snapshot||{},p=s.scene_probs||{default:0,fire:0,smoke:0},st=s.alarm_state||'normal',c=alarmColor(st);$('alarm').textContent=st.toUpperCase();$('alarm').style.background=c[0];$('alarm').style.color=c[1];$('pipeTag').textContent=d.current_model_key?`${s.model||d.current_model_key} · Scene ${d.scene_enabled?'ON':'OFF'} / ${d.scene_interval}`:'WAITING';$('fps').textContent=fmt(s.fps);$('avg').textContent=fmt(s.avg_fps);$('yms').textContent=fmt(s.yolo_inference_ms);$('pms').textContent=fmt(s.pipeline_ms);$('fc').textContent=fmt(s.max_fire_conf,3);$('sc').textContent=fmt(s.max_smoke_conf,3);$('sd').textContent=fmt(p.default,3);$('sf').textContent=fmt(p.fire,3);$('ss').textContent=fmt(p.smoke,3);let b=Object.entries(p).sort((a,b)=>b[1]-a[1])[0];$('sceneCls').textContent=b?b[0]:'default';$('sceneMode').textContent=!d.scene_enabled?'OFF':(s.scene_updated?'REFRESHED':'CACHED');$('fst').textContent=`${s.fire_streak||0} / 3`;$('sst').textContent=`${s.smoke_streak||0} / 5`;$('safe').textContent=`${s.safe_streak||0} / 10`;$('reason').textContent=s.alarm_reason||'—';$('gpu').textContent=d.system.gpu||'—';$('rss').textContent=`${fmt(s.rss_mb,0)} MB`;$('cmem').textContent=`${fmt(s.cuda_allocated_mb,0)} MB`;let mi=s.model_info||{};$('msize').textContent=mi.model_size_mb==null?'—':`${fmt(mi.model_size_mb,1)} MB`;$('params').textContent=mi.parameters==null?'—':Number(mi.parameters).toLocaleString();$('gflops').textContent=mi.gflops==null?'—':fmt(mi.gflops,2);$('run').textContent=d.run_dir||'—';$('error').textContent=d.error||'';let run=!!d.running;['model','scene','sceneOn','interval'].forEach(id=>$(id).disabled=run);$('start').disabled=run;$('pause').disabled=!run;$('stop').disabled=!run;$('pause').textContent=d.paused?'Resume':'Pause';let z=d.series||{};chart('fpsChart',[{n:'FPS',v:z.fps||[]}]);chart('latChart',[{n:'YOLO ms',v:z.yolo_ms||[]},{n:'Pipeline ms',v:z.pipeline_ms||[]}])}catch(e){$('error').textContent='Backend disconnected.'}}
async function start(){let body={model_key:$('model').value,scene_model_key:$('scene').value,scene_enabled:$('sceneOn').checked,scene_interval:Number($('interval').value),source:$('source').value,duration:Number($('duration').value||0),test_id:$('testid').value||'test01',save_video:$('videoOn').checked};let r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();if(!r.ok)alert(d.error||'Start failed');refresh()}
async function hist(){let d=await(await fetch('/api/history',{cache:'no-store'})).json();$('tbody').innerHTML=(d.rows||[]).slice(0,12).map(x=>`<tr><td>${x.model||''}</td><td>${fmt(x.avg_fps)}</td><td>${fmt(x.mean_yolo_inference_ms)}</td><td>${fmt(x.mean_pipeline_ms)}</td><td>${fmt(x.mean_rss_mb,0)}</td><td>${x.mean_alarm_response_s==null?'—':fmt(x.mean_alarm_response_s)}</td><td>${x.scene_enabled===false?'OFF':'ON'}</td><td>${x.scene_interval??'—'}</td></tr>`).join('');let a=d.latest_by_model||[],mx=Math.max(1,...a.map(x=>Number(x.avg_fps||0)));$('bars').innerHTML=a.length?a.map(x=>`<div class="barrow"><div>${x.model}</div><div class="track"><div class="fill" style="width:${Math.min(100,Number(x.avg_fps||0)/mx*100)}%"></div></div><div style="text-align:right">${fmt(x.avg_fps)} FPS</div></div>`).join(''):'<div class="note">Run each detector once to populate the comparison.</div>'}
$('start').addEventListener('click',start);$('pause').addEventListener('click',()=>fetch('/api/pause',{method:'POST'}));$('stop').addEventListener('click',()=>fetch('/api/stop',{method:'POST'}));window.addEventListener('resize',refresh);models();hist();refresh();setInterval(refresh,500);setInterval(hist,5000);
</script></body></html>
"""

@app.route('/')
def index():return render_template_string(HTML)

@app.route('/api/models')
def api_models():
    d,s=discover_models();return jsonify(detectors=d,scenes=s)

@app.route('/api/status')
def api_status():
    with lock:return jsonify(json.loads(json.dumps(STATE,default=str)))

@app.route('/api/history')
def api_history():return jsonify(history_payload())

@app.route('/api/start',methods=['POST'])
def api_start():
    global worker_thread,STATE,latest_jpeg
    with lock:
        if STATE['running']:return jsonify(ok=False,error='Stop the current run before switching/loading another model.'),409
    data=request.get_json(silent=True) or {};dets,scenes=discover_models();dm={x['key']:x for x in dets};sm={x['key']:x for x in scenes};mk=str(data.get('model_key',''));sk=str(data.get('scene_model_key',''));scene_on=bool(data.get('scene_enabled',True))
    if mk not in dm:return jsonify(ok=False,error='Selected YOLO model not found under ./models'),400
    if scene_on and sk not in sm:return jsonify(ok=False,error='Scene Assistance is ON but no valid scene model is selected.'),400
    item=dm[mk];scene=sm.get(sk)
    cfg=dict(DEFAULT);cfg.update(model_key=mk,model_label=item['label'],model_path=item['path'],scene_model_path=scene['path'] if scene else '',scene_enabled=scene_on,scene_interval=int(data.get('scene_interval',3)),source=str(data.get('source','0')),duration=float(data.get('duration',0)),test_id=str(data.get('test_id','test01')) or 'test01',save_video=bool(data.get('save_video',True)))
    with lock:
        system=STATE['system'];STATE=fresh_state();STATE['system']=system;STATE.update(current_model_key=mk,current_model_path=item['path'],scene_model_path=cfg['scene_model_path'],scene_enabled=scene_on,scene_interval=cfg['scene_interval'],status='Starting')
    latest_jpeg=None;stop_event.clear();pause_event.clear();worker_thread=threading.Thread(target=worker,args=(cfg,),daemon=True);worker_thread.start();return jsonify(ok=True)

@app.route('/api/pause',methods=['POST'])
def api_pause():
    with lock:
        if not STATE['running']:return jsonify(ok=False,error='No active run'),409
    pause_event.clear() if pause_event.is_set() else pause_event.set();return jsonify(ok=True)

@app.route('/api/stop',methods=['POST'])
def api_stop():stop_event.set();return jsonify(ok=True)

@app.route('/video_feed')
def video_feed():
    ph=placeholder()
    def gen():
        last=None
        while True:
            j=latest_jpeg or ph
            if j!=last:last=j;yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+j+b'\r\n'
            time.sleep(.03)
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--host',default='0.0.0.0');ap.add_argument('--port',type=int,default=5000);args=ap.parse_args();RESULTS_ROOT.mkdir(parents=True,exist_ok=True)
    d,s=discover_models();print('\n=== Jetson Fire/Smoke Dynamic Alarm ===');print('Jetson:',jetson_model());print('GPU:',gpu_name());print('Torch:',torch.__version__,'CUDA:',torch.version.cuda,'available=',torch.cuda.is_available());print('Ultralytics:',ultralytics.__version__);print('OpenCV:',cv2.__version__);print('\nYOLO models:');[print(' -',x['label'],'->',x['path']) for x in d];print('Scene classifiers:');[print(' -',x['label'],'->',x['path']) for x in s];print(f'\nOpen: http://127.0.0.1:{args.port}\n');app.run(host=args.host,port=args.port,threaded=True,use_reloader=False)

if __name__=='__main__':main()
