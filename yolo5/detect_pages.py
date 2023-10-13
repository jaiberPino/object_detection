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
co_t=[]
mp=[]
@smart_inference_mode()
def run(
        weights= None,  # model.pt path(s)
        img=None,
        data=ROOT / 'utils/custom_data.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.50,  # confidence threshold
        iou_thres=0.75,  # NMS IOU threshold 
        max_det=30,  # maximum detections per image
        classes=0,  # filter by class: --class 0, or --class 0 2 3
        name='pages',  # save results to project/name
):
    webcam = False
    # Load model
    device = select_device('cpu')
    model = weights
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    bs = 1 # batch_size
    # Dataloader
    if webcam == False and name == 'pages': #aqui entra para extraer el doc donde detecta pages
        dataset = LoadImages('nothing', img_size=imgsz, stride=stride, auto=pt, vid_stride=1, sel=name, coorT=None, imagen=img)#carga la imagen
    #  Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt = (Profile(), Profile(), Profile())
    score=[]
    cord=[]
    h=0
    for path, im, im0s, vid_cap, s , ya, angulo in dataset:
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
            
            if name=='pages':
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
                if name =='pages':
                    scr=det.tolist()
                    k=0
                    for s in scr:
                        for sco in score:
                            if s[4]==sco:
                                for *xyxy, conf, cls in reversed([scr[k]]):
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                    h,w,_=im0.shape
                                    x1 = xywh[0]*w - ((xywh[2]*w)/2) 
                                    y1=  xywh[1]*h -((xywh[3]*h)/2)
                                    x2= xywh[2]*w+x1
                                    y2=  xywh[3]*h + y1
                                    cord.append([int(x1), int(y1), int(x2), int(y2)])
                        k=k+1
    print("SCORE PAGES: ",score)  
    return cord

def main(flag='pages', img=None,size=(640,640),model1=None):
    #check_requirements(exclude=('tensorboard', 'thop'))
    if flag == 'pages':
        c =run(weights = model1, img=img, imgsz=size, conf_thres=0.7)
    return c

def ppal(img,h,w, model_page): 
    #inferencia tabla
    inicio=time.time()
    c_p=main(img=img, size=(h,w), model1 = model_page)
    print("COORDENADAS PAGES ",c_p)
    if c_p != []:
        pass
    fin=time.time()
    print("TOTAL EXECUTION TIME ",fin-inicio)
    return c_p#  coord_rows, coord_col, coord_table,  score_rows, score_ col, score_table
    
def pages_detector(im2,w,h,model_page):
    out = ppal(im2, h ,w, model_page)
    if out is not None:
        coor_detect=out
    else:
        coor_detect=None
    return coor_detect
 
   
#imh=cv2.imread(ROOT / 'images/test_reading28.jpeg')
#w,h,_=imh.shape
#col_row(imh,640,640)