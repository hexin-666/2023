from sklearn.metrics.cluster import  mutual_info_score
import numpy as np
import cv2
path1='Fusion_results/3.jpg'
path2='Fusion_results/2.jpg'
img1=cv2.imread(path1)
img2=cv2.imread(path2)
img1 = cv2.resize(img1,(141,329))
img_ref = np.array(img1, dtype=np.int32)/255
img_sen = np.array(img2, dtype=np.int32)/255

img_ref=img_ref .reshape(-1)
img_sen_roi=img_sen .reshape(-1)
MIValue=mutual_info_score(img_ref, img_sen_roi)
print('MI',MIValue)
