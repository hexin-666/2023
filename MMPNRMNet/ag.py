import os

import numpy as np
import math
from PIL import Image


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


if __name__ == '__main__':
    path = r"D:\MMPNRMNet\image_out"
    filelist = os.listdir(path)
    ags = 0
    for file in filelist:
        ag = avgGradient(os.path.join(path, file))
        print("ag:{}".format(ag))
        ags = ags + ag
    print("average",ags/5)
