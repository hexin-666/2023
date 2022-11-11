import os

import cv2
import numpy as np
import math

import skimage.measure



path = r'D:\MMPNRMNet\image_out'
path1 = r'D:\MMPNRMNet\data\train\GT'

filelist = os.listdir(path)
sfs = 0
for file in filelist:
    img = cv2.imread(os.path.join(path,file),0)
    img_fu = cv2.imread(os.path.join(path1,file),0)

    ssim = skimage.measure.compare_ssim(img, img_fu, data_range=255)
    print("ssim",ssim)

    psnr = skimage.measure.compare_psnr(img, img_fu, data_range=255)
    print("psnr",psnr)

    mse = skimage.measure.compare_mse(img, img_fu)
    print("mse",mse)

    rmse = math.sqrt(mse)
    print("rmse",rmse)

    nrmse = skimage.measure.compare_nrmse(img, img_fu, norm_type='Euclidean')
    print("nrmse",nrmse)

    entropy = skimage.measure.shannon_entropy(img_fu, base=2)
    print("entropy",entropy)




