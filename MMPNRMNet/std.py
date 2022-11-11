import os

import cv2
import numpy as np
from PIL import Image


def STD(pic_path):
    image = Image.open(pic_path).convert('L')
    image = image.resize((141, 329))
    # image = cv2.resize(image, (141, 329),interpolation=cv2.INTER_CUBIC)
    image = np.array(image) / 255
    (mean, stddv) = cv2.meanStdDev(image)

    return round(stddv[0][0], 4)  # 84.7159


if __name__ == '__main__':
    path = r'D:\MMPNRMNet\image_out'
    path1 = r'D:\MMPNRMNet\data\train\GT'
    path2 = r'D:\MMPNRMNet\data1\train\GT'
    filelist = os.listdir(path)
    stds = 0
    for file in filelist:
        std = STD(os.path.join(path, file))
        print("avg_std{}".format(std))
        stds = stds + std
    print(stds/5)
