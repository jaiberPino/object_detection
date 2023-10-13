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
from utils.yolo5 import detect2  
from utils.yolo5 import detect_pages
from MedicalNLU.GoogleJsonOperations import get_orientation, rotate_vertices

def get_words(data, width, height):
    words = []
    orientation=0

    for block in data['pages'][0]['blocks']:
        for paragraph in block['paragraphs']:
            for word in paragraph['words']:
                word['boundingBox'] = denormalize(word['boundingBox'], width, height, orientation)
                words.append(word)

    return words
    
def scale_vertice(vertice, width, height):
    retv = {'x': vertice['x']*width, 'y': vertice['y']*height}
    return retv
    
def rotate_point(point, orientation, width, height):
    ox, oy = point

    retx = ox
    rety = oy
    if orientation == 270:
        retx = height - oy
        rety = ox
    elif orientation == 180:
        retx = width - ox
        rety = height - oy
    elif orientation == 90:
        retx = oy
        rety = height - ox
    
    if retx < 0:
        retx = 0
    if rety < 0:
        rety = 0
    return retx, rety
    
def rotate_vertice(vertices, orientation, width, height):
    for vertice in vertices:
        point = (vertice['x'], vertice['y'])
        vertice['x'], vertice['y'] = rotate_point(point, orientation, width, height)

    return vertices
    
def denormalize(bound, width, height, orientation):
    try:
      if len(bound['vertices']) > 0:
          vertices = bound['vertices']
      p=vertices
    except UnboundLocalError:
      if len(bound['normalizedVertices']) > 0:
          vertices = []
          for vertice in bound['normalizedVertices']:
              vertices.append(scale_vertice(vertice, width, height))
  
          #vertices = rotate_vertice(vertices, orientation, width, height)
      elif len(bound['vertices']) > 0:
        vertices = bound['vertices']
    return { 'vertices' : vertices }
    
def pendiente(x1,y1,x2,y2): 
        if (x2-x1) != 0: 
          m = (y2-y1)/(x2-x1) 
        else: 
          m = 0 
        b = y2 - m*x2 
        return m,b 
        
def point_y_in_line(x,m,b): 
      y= m*x + b 
      return y
      
def middle_point(vertices): 
    new_x = vertices[0]['x'] + (vertices[1]['x']-vertices[0]['x'])/2 
    m,b=pendiente(vertices[0]['x'],vertices[0]['y'],vertices[1]['x'],vertices[1]['y']) 
    m2,b2=pendiente(vertices[3]['x'],vertices[3]['y'],vertices[2]['x'],vertices[2]['y']) 
    y_up= point_y_in_line(new_x,m,b) 
    y_dow= point_y_in_line(new_x,m2,b2) 
    new_y = y_up + (y_dow-y_up)/2 
    #point={'x': new_x, 'y': new_y} 
    point=(new_x, new_y)
    return point
    
def middle_point2(vertices):
  new_x = vertices[0]['x'] + (vertices[1]['x']-vertices[0]['x'])/2 
  if vertices[0]['y'] >= vertices[3]['y']:
    new_y = vertices[3]['y'] + (abs(vertices[3]['y']-vertices[0]['y']))/2 
  else:
    new_y = vertices[0]['y'] + (abs(vertices[3]['y']-vertices[0]['y']))/2
  point=(new_x, new_y)
  return point
  
def middle_point3(vertices):
  new_x = vertices[3]['y'] + (vertices[3]['y']-vertices[0]['y'])/2 
  if vertices[0]['y'] >= vertices[3]['y']:
    new_y = vertices[0]['x'] + (abs(vertices[1]['x']-vertices[0]['x']))/2 
  else:
    new_y = vertices[0]['x'] + (abs(vertices[1]['x']-vertices[0]['x']))/2
  point=(new_x, new_y)
  return point
  
def get_centroid_from_words(words, w, h, data_j):#cambios aqui
    words = get_words(data_j, w, h)   
    cent = []
    ym=0
    xm=0
    try:
      ang=90
      for word in words:
        if ang ==90:
          pto_med = middle_point2(word['boundingBox']['vertices'])
          ym=ym+word['boundingBox']['vertices'][3]['y'] - word['boundingBox']['vertices'][0]['y']
          xm=xm+word['boundingBox']['vertices'][1]['x'] - word['boundingBox']['vertices'][0]['x']
          cent.append(pto_med)
        elif ang == 0:
          pto_med = middle_point3(word['boundingBox']['vertices'])
          ym=ym+word['boundingBox']['vertices'][1]['x'] - word['boundingBox']['vertices'][0]['x']
          xm=xm+word['boundingBox']['vertices'][3]['y'] - word['boundingBox']['vertices'][0]['y']
          cent.append(pto_med)
        else:
          pass
      value = (ym/len(words))/3
      if w < 1000 and h < 1000:
        valuex = (xm/len(words))/4
        value = (ym/len(words))/6
      else:
        valuex = (xm/len(words))/1#2
      
    except IndexError:
      for word in words:
        x1 = word['boundingBox']['normalizedVertices'][0]['x'] * w
        x2 = word['boundingBox']['normalizedVertices'][1]['x'] * w
        y1 = word['boundingBox']['normalizedVertices'][0]['y'] * h
        y2 = word['boundingBox']['normalizedVertices'][3]['y'] * h
        pto_med=(((x2-x1)/2)+x1, ((y2-y1)/2)+y1) #punto medio (x,y)
        cent.append(pto_med)

    return cent,value, valuex
    
def get_text_from_words(words):
    text = ''
    try:
      try:
        for word in words:
          for symbol in word['symbols']: #GOOGLE EXTRAE TEXTO PROCEDIMINETOS opcion 1
            text = text+symbol['text']
          text = text + ' '
      except TypeError:
        for word in words:
          for w in word[0]['symbols']:#GOOGLE EXTRAE TEXTO PROCEDIMINETOS opcion 2
            try:
              text = text+w['text']
            except KeyError:
              text = text+w[0]['text']
          text = text + ' '
          
    except (TypeError, KeyError):
      #AZURE
      for word in words:
          if type(word) == dict:
            text = text + ' '+word['text'] 
          else:
            if len(word)>1:
              for t in word:
                text = text+' '+t['text']
            else:
              text = text + ' '+word[0]['text']
    return text
    
def get_line_segments(words):
    segments = []
    for word in words:
        x0 = (word['boundingBox']['vertices'][3]['x'] + word['boundingBox']['vertices'][0]['x'])/2
        y0 = (word['boundingBox']['vertices'][3]['y'] - word['boundingBox']['vertices'][0]['y'])/2 + word['boundingBox']['vertices'][0]['y']
        x1 = (word['boundingBox']['vertices'][2]['x'] + word['boundingBox']['vertices'][1]['x'])/2
        y1 = (word['boundingBox']['vertices'][2]['y'] - word['boundingBox']['vertices'][1]['y'])/2 + word['boundingBox']['vertices'][1]['y']
        m, b = get_line_params(x0, y0, x1, y1)
        s = [x0, y0, x1, y1, m, b]
        segments.append(s)
    return segments
  
def get_line_params(x0, y0, x1, y1):
    #y = mx + b
    m = (y1 - y0) / (x1 - x0+0.0001)
    b = y1 - m * x1
    return m, b
    
def merge_segments(lines, val, valx):
    def get_lines_by_separation(line, lines, max_x_sep=valx, max_y_sep=val):#90
        inside = [line]
        outside = []
        for l in lines:
            x_sep = abs(line[2] - l[0])
            y_sep = abs(line[3] - l[1])
            if (x_sep <= max_x_sep) and (y_sep <= max_y_sep):
                inside.append(l)
            else:
                outside.append(l)

        return inside, outside

    def merge_lines(lines):
        m = mean([l[4] for l in lines])
        if m > 0:
            x0 = min(lines, key=lambda l: l[0])[0]
            y0 = min(lines, key=lambda l: l[1])[1]
            x1 = max(lines, key=lambda l: l[2])[2]
            y1 = max(lines, key=lambda l: l[3])[3]
        else:
            x0 = min(lines, key=lambda l: l[0])[0]
            y0 = max(lines, key=lambda l: l[1])[1]
            x1 = max(lines, key=lambda l: l[2])[2]
            y1 = min(lines, key=lambda l: l[3])[3]

        m, b = get_line_params(x0, y0, x1, y1)
        s = [x0, y0, x1, y1, m, b]
        return s

    keep = lines.copy()
    new_lines = []
    head = keep.pop(0)

    while True:
        merge, keep = get_lines_by_separation(head, keep)
        while len(merge) > 1:
            head = merge_lines(merge)
            merge, keep = get_lines_by_separation(head, keep)
        else:
            new_lines.append(head)

        if len(keep) == 0:
            break
        
        head = keep.pop(0)

    return new_lines

def sort_lines_by_height(lines, width, height):
    avgm = mean([l[4] for l in lines])

    if avgm > 0:
        sorted_lines = sorted(lines, key=lambda l: l[1])
    else:
        sorted_lines = sorted(lines, key=lambda l: height - l[3], reverse=True)
    
    return sorted_lines
    
def text_line(word, lineas,w,h,val,valx):
  segments = get_line_segments(word)
  m_lines = merge_segments(segments, val, valx)
  sorted_merged_lines = sort_lines_by_height(m_lines, w, h)# aqui va w, h
  words_by_lines = get_words_over_lines(word, sorted_merged_lines)
  ordered_text = get_text_from_words(words_by_lines)
  return ordered_text, words_by_lines
    
def cargar_datos(m, rt):
  for h in m:  #azure
    if len(h)>0:
      rt.append(h)
  
  return rt 
  
def extract_text(datos,resultado, sorted_words, new_lines,w,h,val,valx, ind_all):
  boxx=[]
  box=[]
  resul=[]
  
  index_table1=[]
  d_celda=None
  if resultado != []:
    for item in resultado:
      if item not in resul:
          resul.append(item)
    for cn in datos:      
      if cn[1] not in ind_all or ind_all ==[]:
        for celda in resul:
          if celda==cn[0]:
            if d_celda is None:
              box.append(sorted_words[cn[1]])
              d_celda=celda
            elif celda==d_celda:
              box.append(sorted_words[cn[1]])
              d_celda=celda
            else:
              boxx.append(box)
              box = []
              box.append(sorted_words[cn[1]])
              d_celda=celda
   
      index_table1.append(cn[1]) 
    if box !=[]:
        boxx.append(box)
    result_line=[]
    for cels in boxx:
      result_line.append(text_line(cels,new_lines,w,h,val,valx))
    ls=len(sorted_words)
    t_pro1=[]
    text_sort1 =""
    for datos in result_line:
      t_pro1=cargar_datos(datos[1], t_pro1)
      text_sort1 = text_sort1 + datos[0]
      
  else:
    print("VACIO")
  return text_sort1, index_table1, ls, t_pro1

def filas_columnas(pto,j,ubi,ubi2,coor_col,save):
  aux=False
  
  for cc in coor_col:#analiza las columnas
    flag2=puntoEnPoligono(cc,pto)
    aux=aux+flag2
    if (flag2):
      ubi.append((cc[0]['x'] + cc[1]['x'],j))
      ubi2.append(cc[0]['x'] + cc[1]['x'])
      ubi2.sort()
      result = []
      for item in ubi2:
          if item not in result:
              result.append(item)
  if aux==0:
    result=save
  save=result
  return(ubi,result,save)
  
def is_line_over_bbox(bbox, line):#
    #y = mx + b
    m = (line[3] - line[1]) / (line[2] - line[0]+0.001)
    b = line[3] - m * line[2]
    if (bbox['vertices'][0]['y'] <= m * bbox['vertices'][0]['x'] + b <= bbox['vertices'][3]['y'] and 
        bbox['vertices'][1]['y'] <= m * bbox['vertices'][1]['x'] + b <= bbox['vertices'][2]['y']):
        ret = True
    else:
        ret = False
    return ret
    
def get_words_over_lines(words, lines):
    def get_words_by_line(words, line):
        words_by_line = []
        for word in words:
            if is_line_over_bbox(word['boundingBox'], line):
                words_by_line.append(word)
        
        return words_by_line

    words_by_lines = []
    cwords = words.copy()

    for line in lines:
        words_by_line = get_words_by_line(cwords, line)
        sorted_words_by_line = sorted(words_by_line, key=lambda w: w['boundingBox']['vertices'][0]['x'])
        words_by_lines.extend(sorted_words_by_line)
        cwords = [word for word in cwords if word not in words_by_line]

    return words_by_lines
    
    
def orientacion(p1,p2,p3):
 return (p2['x']-p1[0])*(p3['y']-p1[1])-(p2['y']-p1[1])*(p3['x']-p1[0]) >0
 
def puntoEnPoligono(tabla,punto):
  orientacion1= orientacion(punto,tabla[0],tabla[1])
  orientacion2= orientacion(punto,tabla[1],tabla[2])
  orientacion3= orientacion(punto,tabla[2],tabla[3])
  orientacion4= orientacion(punto,tabla[3],tabla[0])

  return (orientacion1 and orientacion2 and orientacion3 and orientacion4) or (not(orientacion1) and not(orientacion2) and not(orientacion3) and not(orientacion4))

def proccess(data, coor_R, coor_col, width, h, coor_table):
    words = get_words(data, width, h) #text, confidence, boundingBox:vertices
    centros,val, valx = get_centroid_from_words(words, width, h,data) #centroides de las palabras
    ubi=[]
    save=[]
    ubi_col_tex=[]
    new_lines=[]
    
    #LINEAS que estan en la tabla
    segments = get_line_segments(words)
    m_lines = merge_segments(segments, val, valx)
    
    sorted_lines = sort_lines_by_height(m_lines, width, h)
    for lin in sorted_lines:
      if (width / (coor_table[0][3] - coor_table[0][1]+0.01)) > 1 or (width / (coor_table[0][3] - coor_table[0][1]))<=1:
        if lin[1]>coor_table[0][1] and lin[1]<coor_table[0][3]:
          new_lines.append(lin)
      else:
        if lin[1]>coor_table[0][1] and lin[1]<coor_table[0][3]:
          new_lines.insert(0,lin)
    #FILAS VS COLUMNAS
    j=0
    fila=[]
    ptos_t=[]
    for cr in coor_R:
      for pto in centros: 
        flag=puntoEnPoligono(cr,pto)
        if (flag):
          if pto not in ptos_t:
            fila,ubi,save=filas_columnas(pto,j,fila,ubi,coor_col,save)
            ptos_t.append(pto)
        j +=1
      fila.sort()
      ubi_col_tex.append(fila)
      fila=[]
      j=0
    return(ubi_col_tex, ubi, words, new_lines, val, valx)    
    
def join_text(data,coor_R, coor_col, wi, he, coor_table):
    if coor_R ==[] or coor_col==[]:
      print("NO DETECTO TABLA O FILAS O COLUMNAS")
      rest_text = []
      all_text = ''
    else:
      index_all=[]
      words = get_words(data,wi, he)
      #print("TAMAÑO INICIAL",len(words))
      ls=len(words)
      index=[0]
      text_s, ubi, sw, n_l, val, valx = proccess(data,coor_R,coor_col, wi, he, coor_table)
      #print("VALX",wi,he,valx,val)
      text_table=[]
      for t in text_s:
        if t!=[]:
          tx, index, ls, t_pr = extract_text(t, ubi, sw, n_l,wi,he,val,valx, index_all)
          print(tx)
          for ind in index:
            if ind not in index_all:
              index_all.append(ind)
          text_table.append(t_pr)#aqui queda el texto de la tabla
      li=len(index_all)  
      if index_all !=[]:
        mn=np.min(index_all)
      else:
        mn=0
      rest_text=[]
      for n in range(0,ls,1):
        if n not in index_all:
          rest_text.append(sw[n])
      kk=len(rest_text)
      p=0
      for tx in range(0, ls ,1):
        if tx == mn:
          i=mn
          for fila in text_table:
            for fn in fila:
              rest_text.insert(i,fn)
              i=i+1
              p=p+1
      all_text=get_text_from_words(rest_text) 
      #print("ALL TEXT")
      print("CANTIDAD DE WORDS ORIGINAL:",len(words),"CANTIDAD DE WORDS FINAL:",len(rest_text),"words in table:",len(index_all),"words out table:",kk,"total words in table:",p)
      completed_text = True
      if ls+10 <= len(rest_text) or ls-3 >= len(rest_text):
        completed_text = False
      #print(all_text)
    return(rest_text, all_text, completed_text)
    
def ajuste_coor(coor_act, coor_p):
  coor_n =  [{'x': 0, 'y': 0}, {'x': 0, 'y': 0}, {'x': 0, 'y': 0}, {'x': 0, 'y': 0}]
  coor_n[0]['x'] = coor_act[0]['x'] + coor_p[0]
  coor_n[1]['x'] = coor_act[1]['x'] + coor_p[0]
  coor_n[2]['x'] = coor_act[2]['x'] + coor_p[0]
  coor_n[3]['x'] = coor_act[3]['x'] + coor_p[0]
  coor_n[0]['y'] = coor_act[0]['y'] + coor_p[1]
  coor_n[1]['y'] = coor_act[1]['y'] + coor_p[1]
  coor_n[2]['y'] = coor_act[2]['y'] + coor_p[1]
  coor_n[3]['y'] = coor_act[3]['y'] + coor_p[1]
  return coor_n
  
def page_detect(img, pages_coor, model_tab, model_rows, model_col):
  rows_c = []
  rows_s = []
  col_c = []
  col_s = []
  
  for pc in pages_coor:
    img1=img[pc[1]:pc[3],pc[0]:pc[2]]
    coor_All=detect2.col_row(img1,640,640,model_tab,model_rows,model_col)
    for r in coor_All['rows']:
      a = ajuste_coor(r,pc)
      rows_c.append(a)
    for r in coor_All['col']:
      a = ajuste_coor(r,pc)
      col_c.append(a)
    for r in coor_All['score_rows']:
      rows_s.append(r)
    for r in coor_All['score_col']:
      col_s.append(r)
    xi=4000
    ysu = 4000
    xd = 0
    yi = 0
    for r in coor_All['table']:
      if r[0]<xi:
        xi=r[0]
      if r[2] > xd:
        xd = r[2]
      if r[1] < ysu:
        ysu = r[1]
      if r[3] > yi:
        yi = r[3]

  coor_All['rows'] = rows_c
  coor_All['score_rows'] = rows_s
  coor_All['col'] = col_c
  coor_All['score_col'] = col_s
  coor_All['table'] = [xi,ysu,xd,yi]
  return(coor_All)
    
def all_rotations(data_json, img, w, h):
    angle=get_orientation(data_json)
    if angle > 0:
      if len(data_json['pages'][0]['blocks']) <= 1:# AZURE - rotacion de las bbox
        for d in data_json['pages'][0]['blocks'][0]['paragraphs'][0]['words']:
          v=(d['boundingBox']['vertices'])
          rotate_vertices(v,(w,h), 360-angle)
      else:
        for d in data_json['pages'][0]['blocks']: #GOOGLE - rotacion de la bbox
          for m in d['paragraphs']:
            for b in m['words']:
              v=b['boundingBox']['vertices']
              if v == []:
                for vert in b['boundingBox']['normalizedVertices']:
                  v.append({'x':vert['x']* w, 'y':vert['y']* h })
              rotate_vertices(v,(w,h), 360-angle)
      
    if angle == 90:
      image=np.rot90(img, k=1)  
    if angle == 180:
      image=np.rot90(img, k=2)
    if angle == 270:
      image=np.rot90(img, k=3)
    if angle == 0:
      image = img
    return image, angle
    
def save_json(data, json_copy, all_text, only_text,coor_All,orientation, w, h):
    #ADJUNTAR EL NUEVO READING ORDER AL JSON
    data['pages'][0]['block2']=json_copy['pages'][0]['blocks']
    all_coor_col = []
    all_coor_row = []
    coor_table= [{'x':coor_All['table'][0][0], 'y':coor_All['table'][0][1]}, {'x':coor_All['table'][0][2], 'y':coor_All['table'][0][1]}, {'x':coor_All['table'][0][2], 'y':coor_All['table'][0][3]}, {'x':coor_All['table'][0][0], 'y':coor_All['table'][0][3]}]
    data['pages'][0]['table_data'] ={'tables':[{'vertices' : coor_table, 'score':coor_All['score_table']}]}
    m=0
    for i in coor_All['rows']:
      all_coor_row.append({'vertices':i,'score':coor_All['score_rows'][m]})
      m=m+1
    m=0
    
    for i in coor_All['col']: 
      all_coor_col.append({'vertices':i,'score':coor_All['score_col'][m]})
      m=m+1
    data['pages'][0]['table_data']['columns'] = all_coor_col
    data['pages'][0]['table_data']['rows'] = all_coor_row  
    dic_text={'paragraphs':[{'words':all_text}]}
    try:
      ex = data['pages'][0]['blocks'][1]
      data['pages'][0]['blocks']=['a']
      dic_text={'paragraphs':[{'words':all_text}]}
      data['pages'][0]['blocks'][0]=dic_text # borrar
    except IndexError:
      data['pages'][0]['block2']=json_copy['pages'][0]['blocks'] # borrar despues de pruebas
      data['pages'][0]['blocks'][0]=dic_text # borrar
    data['pages'][0]['orientation']= orientation
    data['pages'][0]['is_rotated']= True
    #print("ANG",orientation)
    if orientation == 90 or orientation == 270:
      data['pages'][0]['width'] = h
      data['pages'][0]['height'] = w
    data['text_sort']=only_text
    all_text=[]
    only_text=[]
    dic_text=[]
    return data

def reduce_dim(hj, wj, img, ang, change, cord = None):
  if hj > wj:
    factor_h = hj/1920
    factor_w = wj/1440
  else:
    factor_h = hj/1440
    factor_w = wj/1920
  if change == 0:
    if ang == 90 or ang == 270:
      img_2 = cv2.resize(img, dsize=(int(hj/factor_h), int(wj/factor_w)), interpolation=cv2.INTER_CUBIC)
    else:
      img_2 = cv2.resize(img, dsize=(int(wj/factor_w), int(hj/factor_h)), interpolation=cv2.INTER_CUBIC)
  else:
    if ang == 90 or ang == 270:
      factor_w1 = factor_h
      factor_h1 = factor_w      
    else:
      factor_w1 = factor_w
      factor_h1 = factor_h    
    img_2 = None
    if cord != None:
      for r in cord['rows']:
        for r1 in r:
          r1['x'] = int(r1['x']*factor_w1)
          r1['y'] = int(r1['y']*factor_h1)
      for r in cord['col']:
        for r1 in r:
          r1['x'] = int(r1['x']*factor_w1)
          r1['y'] = int(r1['y']*factor_h1)
      if cord['table'] != []:
        cord['table']= [[int(cord['table'][0][0]*factor_w1), int(cord['table'][0][1]*factor_h1), int(cord['table'][0][2]*factor_w1), int(cord['table'][0][3]*factor_h1)]]
  return img_2, cord
  
#MAIN----------------------------------------------------------------------------------------------

def reading_inference(image,model_tab,model_rows,model_col,json,model_page):
  json_copy = json.copy()
  w = int(json['pages'][0]['width'])
  h = int(json['pages'][0]['height'])
  
  imag, ang =all_rotations(json,image, w, h)
  img=np.array(imag)
  if img.shape[2]>3:
    img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)
  change = 0
  img, c = reduce_dim(h, w, img, ang, change)  
  #pages_coor = detect_pages.pages_detector(img,640,640,model_page)
  pages_coor = ['a'] #CUANDO VAYA A UTILIZAR LO DE LAS PAGINAS EN UNA FOTO ELIMINO ESTO
  if len(pages_coor) > 1:
    print("CORRDENADAS IMAGEN",pages_coor)
    coor_All = page_detect(img, pages_coor, model_tab, model_rows, model_col)
  else:
    coor_All=detect2.col_row(img,640,640,model_tab,model_rows,model_col)
  if coor_All is not None:
    change = 1
    img2, coor_All = reduce_dim(h, w, img, ang, change, cord = coor_All)
    print("OUT_COOR", coor_All)
    coor_R  = coor_All['rows']
    coor_col = coor_All['col']
    coor_table = coor_All['table']
    if coor_R == [] or coor_col == []:
      return json
    else:
      all_text_data, only_text, completed = join_text(json,coor_R, coor_col, w, h, coor_table)
      if completed:
        json_table = save_json(json, json_copy, all_text_data, only_text,coor_All,ang, w ,h)
        return json_table
      else:
        return json
  
    
    
    #save_json(data_tx, all_texto)