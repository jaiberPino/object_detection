import torch
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from .sigver.preprocessing.normalize import preprocess_signature
from .sigver.featurelearning.models import SigNet
import numpy as np
import cv2

def remove_shadow(img):
  rgb_planes = cv2.split(img)

  result_planes = []
  result_norm_planes = []
  for plane in rgb_planes:
      dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
      bg_img = cv2.medianBlur(dilated_img, 21)
      diff_img = 255 - cv2.absdiff(plane, bg_img)
      norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
      result_planes.append(diff_img)
      result_norm_planes.append(norm_img)
      
  #result = cv2.merge(result_planes)
  result_norm = cv2.merge(result_norm_planes)
  return result_norm

def rubric_comparison(img, model_comparison, device, coor_s):   
  canvas_size = (952, 1360)  # Maximum signature size
  real="/home/ec2-user/environment/DincroML/serverless/utils/yolo5/sigver/data/rubric_test.png"
  scores = []
  for coord in coor_s:
    x1, y1 = coord['vertices'][0]['x'], coord['vertices'][0]['y']
    x2, y2 = coord['vertices'][1]['x'], coord['vertices'][1]['y']
    rubric_test = img[y1:y2, x1:x2]
    # Load and pre-process the signature BASE
    #IN S3: s3://pruebadedatos/rubric_test/images_real_rubrics/test_rub.png
    original = cv2.imread(real)
    original = remove_shadow(original)
    original=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
    w,h= original.shape
    fw=1000/w
    fh =700/h
    original1 = cv2.resize(original,(int(w*fw), int(h*fh)))
    w,h = original1.shape
    processed = preprocess_signature(original1, canvas_size)
    cv2.imwrite("/home/ec2-user/environment/DincroML/serverless/utils/yolo5/sigver/data/pr1.png",processed)
    input = torch.from_numpy(processed).view(1, 1, 150, 220)
    input = input.float().div(255).to(device)
    
    # Load and pre-process the signature SAMPLE
    copia = rubric_test#cv2.imread(test)
    copia = remove_shadow(copia)
    copia=cv2.cvtColor(copia,cv2.COLOR_BGR2GRAY)
    copia1 = cv2.resize(copia, (h,w))
    processed2 = preprocess_signature(copia1, canvas_size)
    cv2.imwrite("/home/ec2-user/environment/DincroML/serverless/utils/yolo5/sigver/data/pr2.png",processed2)
    input2 = torch.from_numpy(processed2).view(1, 1, 150, 220)
    input2 = input2.float().div(255).to(device)
    # Extract features
    with torch.no_grad(): # We don't need gradients. Inform torch so it doesn't compute them
        features = model_comparison(input)
        features2 = model_comparison(input2)
    
    #CALCULA DISTANCIA ENTRE FEATURES
    resultDistance = euclidean_distances(features, features2)
    c=cosine_similarity(features, features2)
    if resultDistance[0][0] <= 15:
      print("FIRMA AUTORIZADA", resultDistance)
    else:
      print("FIRMA FALSA", resultDistance)
    scores.append({'euclidean':resultDistance[0][0], 'cosine':c[0][0]})
  return(scores)