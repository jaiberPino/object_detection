# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import os
import platform
import sys
from pathlib import Path
import time
import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.yolo5.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.yolo5.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.yolo5.torch_utils import select_device, smart_inference_mode

#co_t=[]
#mp=[]
@smart_inference_mode()
def run(
        weights= None,  # model.pt path(s)
        img=None,
        #data=ROOT / 'utils/custom_data.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.50,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold 
        max_det=50,  # maximum detections per image
        classes=0,  # filter by class: --class 0, or --class 0 2 3
        name='table',  # save results to project/name
        coord=None
):
    webcam = False
    co_t = coord
    # Load model
    device = select_device('cpu')
    model = weights
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    bs = 1 # batch_size
    # Dataloader
    if webcam == False and name == 'table': #aqui entra para extraer el doc donde detecta la tabla
        dataset = LoadImages('nothing', img_size=imgsz, stride=stride, auto=pt, vid_stride=1, sel=name, coorT=None, imagen=img)#carga la imagen
    else:
        dataset = LoadImages('nothing', img_size=imgsz, stride=stride, auto=pt, vid_stride=1, sel=name, coorT=coord[0], imagen=img)#carga la imagen
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt = (Profile(), Profile(), Profile())
    score=[]
    cord=[]
    h=0
    
    for path, im, im0s, vid_cap, s , ya, angulo in dataset:
        if angulo != None:
            with dt[0]:
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
    
            pred = model(im, augment=False, visualize=False)
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det)
                pr=pred[0].tolist()
                
                if name=='table':
                    i=[]
                    for p in pr:
                        i.append(p[4])
                    if i!=[]:
                        score=np.max(i)
                else:
                    for p in pr:
                        score.append(p[4])
    
            # Process predictions
            for i, det in enumerate(pred):  # per image
                if webcam == False:  # batch_size >= 1
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
    
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
                    if name =='table':
                        scr=det.tolist()
                        k=0
                        for s in scr:
                            if s[4]==score:
                                for *xyxy, conf, cls in reversed([scr[k]]):
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                    h,w,_=im0.shape
                                    x1 = xywh[0]*w - ((xywh[2]*w)/2) 
                                    y1=  xywh[1]*h -((xywh[3]*h)/2)
                                    x2= xywh[2]*w+x1
                                    y2=  xywh[3]*h + y1
                                    cord.append([int(x1), int(y1), int(x2), int(y2)])
                                co_t=cord
                            k=k+1
                    else:
                        for *xyxy, conf, cls in reversed(det):
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            h,w,_=im0.shape
                            x1 = xywh[0]*w - ((xywh[2]*w)/2) + co_t[0][0]#co_t[0][0]
                            ye=  xywh[1]*h -((xywh[3]*h)/2) 
                            y1=ye + co_t[0][1]
                            x2= xywh[2]*w  + x1 #co_t[0][2]
                            y2=  xywh[3]*h + y1 
                            cord.append([int(x1), int(y1), int(x2), int(y2)])
        else:
            cord=[]
            score=[]
            ya=0
            angulo = 0
    return cord, score, ya, angulo

def main(flag='tables',coor=None, img=None,size=(640,640),model1=None):
    #check_requirements(exclude=('tensorboard', 'thop'))
    if flag == 'tables':
        c,s, ya, ang =run(weights = model1, img=img, coord=coor, imgsz=size )
    if (flag=='rows'):
        c,s, ya, ang =run(weights = model1, img=img, name='row',coord=coor, imgsz=size,conf_thres=0.50,classes=0)
    if (flag=='col'):
        c,s, ya, ang =run(weights = model1, img=img, name='row',coord=coor,imgsz=size,conf_thres=0.40,classes=0)
    return c, s, ya, ang

def ptos_coordenadas(ang, ya, c_r,sel,cor_t):
    if ang<=0:
        ya2=0
        
    else:
        ya2=-ya
        ya=0
        
    rows=[]
    rws=0
    c_r=sorted(c_r, key=lambda x : x[1])
    #EXRAE LOS PUNTOS DE LAS ESQUINAS DE LAS CAJAS
    if sel=='row':
        for i in range(0,len(c_r),1):
            rws=rws+1
            rows.append([{'x':int(c_r[i][0]), 'y':int(c_r[i][1]+ya)},
                            {'x':int(c_r[i][2]+10 ), 'y':int(c_r[i][1]-ya2)},
                            {'x':int(c_r[i][2]+10) , 'y':int(c_r[i][3]-ya2)},
                            {'x':int(c_r[i][0]), 'y':int(c_r[i][3]+ya)}]) 
        print("NUMBER ROWS DETECTIONS= ",rws)
    else:
        for i in range(0,len(c_r),1):
            rws=rws+1
            rows.append([{'x':int(c_r[i][0]), 'y':int(cor_t[0][1])},
                            {'x':int(c_r[i][2]), 'y':int(cor_t[0][1])},
                            {'x':int(c_r[i][2]) , 'y':int(cor_t[0][3])},
                            {'x':int(c_r[i][0]), 'y':int(cor_t[0][3])}]) 
        print("NUMBER COL DETECTIONS= ",rws)
    return rows
def ppal(img,h,w,model_tab,model_rows,model_col):
    #inferencia tabla
    inicio=time.time()
    c_t, sc_t, nada, nada2=main(img=img, size=(h,w),model1=model_tab)
    print("COORDENADAS TABLA, CONFIDENCE",c_t,sc_t)
    if c_t != []:
         #inferencia filas
        c_r, sc_r, ya, ang =main(flag='rows', coor=c_t, img=img, size=(h,w),model1=model_rows)
        rows = ptos_coordenadas(ang, ya, c_r,'row',None)
        
        #inferencia columnas
        c_c, sc_c, ya, ang =main(flag='col', coor=c_t, img=img, size=(h,w),model1=model_col)
        cols = ptos_coordenadas(ang, ya, c_c,'col',c_t)
    else:
        rows=[]
        cols=[]
        sc_c=[]
        sc_r=[]
        sc_t=[]
        c_t=[]
    fin=time.time()
    return rows, cols, c_t,sc_r,sc_c,sc_t #  score tables y filas
    
def col_row(im2,w,h,model_tab,model_rows,model_col):
    out = ppal(im2,h,w,model_tab,model_rows,model_col)
    if out is not None:
        coor_detect={'rows':out[0], 'col':out[1],'table':out[2], 'score_rows':out[3], 'score_col':out[4], 'score_table':out[5]}
    else:
        coor_detect=None
    print("COOR ROWS, COOR COLUMNS",coor_detect)
    return coor_detect
 
   
#imh=cv2.imread(ROOT / 'images/test_reading28.jpeg')
#w,h,_=imh.shape
#col_row(imh,640,640)