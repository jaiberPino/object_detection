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
sys.path.append('/home/ec2-user/environment/DincroML/serverless/')
from utils.yolo5.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.yolo5.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.yolo5.torch_utils import select_device, smart_inference_mode

#co_t=[]
#mp=[]
@smart_inference_mode()
def run(
        weights= None,  # model.pt 
        img=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
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
        dataset = LoadImages('nothing', img_size=imgsz, stride=stride, auto=pt, vid_stride=1, sel=name, coorT=None, imagen=img)#carga la imagen
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
            with dt[1]:
                pred = model(im, augment=False, visualize=False)
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det)
                #pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
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
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        h,w,_=im0.shape
                        x1 = xywh[0]*w - ((xywh[2]*w)/2) 
                        y1=  xywh[1]*h -((xywh[3]*h)/2)
                        x2= xywh[2]*w+x1
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
    if flag == 'logo':
        c,s, ya, ang =run(weights = model1, img=img, name = 'logo', coord=coor, imgsz=size, conf_thres=0.60)
    if (flag=='sign'):
        c,s, ya, ang =run(weights = model1, img=img, name='sign',coord=coor, imgsz=size,conf_thres=0.50)
    return c, s, ya, ang


def ppal(img,h,w,model_l,model_s):
    #inferencia logo y firma
    coor_logo, score_l, nada, nada2=main(flag = 'logo', img=img, size=(h,w),model1=model_l)
    coor_sign, score_s, nada, nada2=main(flag = 'sign', img=img, size=(h,w),model1=model_s)

    return coor_logo,score_l, coor_sign, score_s #  score logo y filas
    
def col_row(im2,w,h,model_logo,model_sign):
    out = ppal(im2,h,w,model_logo,model_sign)
    if out is not None:
        coor_detect={'coor_logo':out[0], 'score_logo':out[1],'coor_sign':out[2], 'score_sign':out[3]}
    else:
        coor_detect=None
    print("DETECTIONS",coor_detect)
    return coor_detect
