import os

import numpy as np
from PIL import Image
import math
import cv2
def SF_evaluation(image):
    M = image.shape[0]
    N = image.shape[1]

    cf =0
    rf =0
    for i in range(1,M-1):
        for j in range(1,N-1):
            dx = float(image[i, j-1])-float(image[i, j])
            rf += dx**2
            dy = float(image[i-1, j])-float(image[i,j])
            cf += dy**2

    RF = math.sqrt(rf/(M*N))
    CF = math.sqrt(cf/(M*N))
    SF = math.sqrt(RF**2+CF**2)
    return SF

if __name__ == '__main__':
    path = r'D:\MMPNRMNet\image_out'
    path1 = r'D:\MMPNRMNet\data\train\GT'
    path2 = r'D:\MMPNRMNet\data1\train\GT'
    filelist = os.listdir(path)
    sfs = 0
    for file in filelist:
        image = Image.open(os.path.join(path,file)).convert('L')
        image = np.array(image) / 255
        sf = SF_evaluation(image)
        print('sf:{}'.format(sf))
        sfs = sfs+sf
    print(sfs/5)

