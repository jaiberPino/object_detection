import json
import numpy as np
import cv2
import os
import boto3
import PIL.Image as Image
import io
import sys 
from statistics import mean
#sys.path.append('/home/ec2-user/environment/DincroML/MedicalNLU/tests/test_jaiber/yolo5_table_row/')#detect2.py
#sys.path.append('/home/ec2-user/environment/DincroML/MedicalNLU/')#GoogleJsonOperation
from utils.yolo5 import detect_logo_sign
from MedicalNLU.GoogleJsonOperations import get_orientation, rotate_vertices
    
def all_rotations(data_json, img, w, h):
    angle=get_orientation(data_json)
    if angle == 90:
      image=np.rot90(img, k=1)  
    if angle == 180:
      image=np.rot90(img, k=2)
    if angle == 270:
      image=np.rot90(img, k=3)
    if angle == 0:
      image = img
    return image, angle
    
def save_json(data, json_copy,coor_All,orientation, w, h, tag_name):
    #ADJUNTAR EL NUEVO READING ORDER AL JSON
    all_coor_logo = []
    all_coor_sign = []
    index_word = len(data['pages'][0]['blocks'][0]['paragraphs'][0]['words'])
    i_w = int(data['pages'][0]['blocks'][-1]['paragraphs'][-1]['words'][-1]['index'])+1
    sym = data['pages'][0]['blocks'][0]['paragraphs'][0]['words'][0].get('symbols', False)
    
    #add logo info to json
    m=0
    for i in coor_All['coor_logo']:
      all_coor_logo.append({'vertices':[{'x':i[0], 'y':i[1]},{'x':i[2], 'y':i[3]}],'score':coor_All['score_logo'][m]})
      if sym:
        data['pages'][0]['blocks'][-1]['paragraphs'][-1]['words'].append({'boundingBox':{'vertices':[{'x':i[0], 'y':i[1]}, {'x':i[2], 'y':i[1]}, {'x':i[2], 'y':i[3]}, {'x':i[0], 'y':i[3]}]} , 'symbols':[{'text':'L'}, {'text':'O'}, {'text':'G'}, {'text':'O'}], 'confidence':coor_All['score_logo'][m],  'index':i_w, 'TAG':tag_name[1], 'score':1})
      else:
        data['pages'][0]['blocks'][0]['paragraphs'][0]['words'].append({'text':'LOGO', 'confidence':coor_All['score_logo'][m], 'boundingBox':{'vertices':[{'x':i[0], 'y':i[1]}, {'x':i[2], 'y':i[1]}, {'x':i[2], 'y':i[3]}, {'x':i[0], 'y':i[3]}]} , 'index':index_word, 'TAG':tag_name[1], 'score':1})
      m=m+1
      index_word =index_word + 1
      i_w = i_w + 1
    
    #add signature info to json
    m=0
    for i in coor_All['coor_sign']:
      all_coor_sign.append({'vertices':[{'x':i[0], 'y':i[1]},{'x':i[2], 'y':i[3]}],'score':coor_All['score_sign'][m]})
      if sym:
        data['pages'][0]['blocks'][-1]['paragraphs'][-1]['words'].append({'boundingBox':{'vertices':[{'x':i[0], 'y':i[1]}, {'x':i[2], 'y':i[1]}, {'x':i[2], 'y':i[3]}, {'x':i[0], 'y':i[3]}]} , 'symbols':[{'text':'R'}, {'text':'U'}, {'text':'B'}, {'text':'R'}, {'text':'I'}, {'text':'C'}], 'confidence':coor_All['score_sign'][m],  'index':i_w, 'TAG':tag_name[0], 'score':1})
      else:
        data['pages'][0]['blocks'][0]['paragraphs'][0]['words'].append({'text':'RUBRIC', 'confidence':coor_All['score_sign'][m], 'boundingBox':{'vertices':[{'x':i[0], 'y':i[1]}, {'x':i[2], 'y':i[1]}, {'x':i[2], 'y':i[3]}, {'x':i[0], 'y':i[3]}]} , 'index':index_word, 'TAG':tag_name[0], 'score':1})
      m=m+1
      index_word =index_word + 1
      i_w = i_w + 1

    res_logo = False
    res_sign = False
    if all_coor_logo != []:
      res_logo = True
    if all_coor_sign != []:
      res_sign = True
    data['pages'][0]['logo'] = {'result':res_logo,'coordinates':all_coor_logo}
    data['pages'][0]['rubric'] = {'result':res_sign, 'coordinates':all_coor_sign}
    return data
      

#MAIN----------------------------------------------------------------------------------------------

def logo_rubric_inference(image,model_l,model_s,json, tag_name):
  json_copy = json.copy()
  w = int(json['pages'][0]['width'])
  h = int(json['pages'][0]['height'])
  
  imag, ang =all_rotations(json,image, w, h)
  img=np.array(imag)
  if img.shape[2]>3:
    img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)
  if ang == 90 or ang == 270:
    img = cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
  else:
    img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

  coor_All=detect_logo_sign.col_row(img,640,640,model_l,model_s)

  if coor_All is not None:
    coor_l  = coor_All['coor_logo']
    coor_s = coor_All['coor_sign']
    json_table = save_json(json, json_copy, coor_All, ang, w, h, tag_name)
    return json_table
  else:
    return json
  