import math
import os
import cv2

import skimage.measure
from PIL import Image
import numpy as np
from skimage.measure import compare_psnr
from sklearn.metrics.cluster import  mutual_info_score

def avgGradient(path):
    image = Image.open(path).convert('L')
    image = image.resize((141, 329))
    image = np.array(image) / 255

    width = image.shape[1]
    width = width - 1
    heigt = image.shape[0]
    heigt = heigt - 1
    tmp = 0.0

    for i in range(heigt):
        for j in range(width):
            dx = float(image[i, j + 1]) - float(image[i, j])
            dy = float(image[i + 1, j]) - float(image[i, j])
            ds = math.sqrt((dx * dx + dy * dy) / 2)
            tmp += ds

    imageAG = tmp / (width * heigt)
    return round(imageAG, 3)



path = r'D:\MMPNRMNet\image_out'
path1 = r'D:\MMPNRMNet\data\train\GT'
path2 = r'D:\MMPNRMNet\data1\train\GT'
filelist = os.listdir(path)
ags = 0
mis = 0
for file in filelist:
    image = cv2.imread(os.path.join(path,file))
    image1 = cv2.imread(os.path.join(path1, file))
    image2 = cv2.imread(os.path.join(path2, file))
    psnr1 = compare_psnr(image, image1, 255)
    psnr2 = compare_psnr(image, image2, 255)
    print("psnr1:",psnr1,"psnr2",psnr2,"Average",(psnr1 + psnr2) / 2)

    image = np.asarray(Image.fromarray(image), dtype=np.float32) / 255.0
    en1 = skimage.measure.shannon_entropy(image,base=2)
    print('en:{}'.format(en1))

    image = cv2.imread(os.path.join(path,file))
    image1 = cv2.imread(os.path.join(path1, file))
    image2 = cv2.imread(os.path.join(path2, file))
    img_ref = np.array(image, dtype=np.int32) / 255
    img_sen1 = np.array(image1, dtype=np.int32) / 255
    img_sen2 = np.array(image2, dtype=np.int32) / 255
    img_ref = img_ref.reshape(-1)
    img_sen_roi1 = img_sen1.reshape(-1)
    img_sen_roi2 = img_sen2.reshape(-1)
    MIValue1 = mutual_info_score(img_ref, img_sen_roi1)
    MIValue2 = mutual_info_score(img_ref, img_sen_roi2)
    mis = mis + MIValue1+MIValue2
    print('MI1', MIValue1,'MI2', MIValue2)
print(mis/5)


