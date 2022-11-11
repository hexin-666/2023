import math
import os

import numpy as np
import cv2
from PIL import Image

L = 1;
Tg = 0.9994;
kg = -15;
Dg = 0.5;
Ta = 0.9879;
ka = -22;
Da = 0.8;


h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)









def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

def convolution(k, data):
    k = flip180(k)
    data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    n,m = data.shape
    img_new = []
    for i in range(n-2):
        line = []
        for j in range(m-2):
            a = data[i:i+3,j:j+3]
            line.append(np.sum(np.multiply(k, a)))
        img_new.append(line)
    return np.array(img_new)


def getArray(img):
    SAx = convolution(h3,img)
    SAy = convolution(h1,img)
    gA = np.sqrt(np.multiply(SAx,SAx)+np.multiply(SAy,SAy))
    n, m = img.shape
    aA = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if(SAx[i,j]==0):
                aA[i,j] = math.pi/2
            else:
                aA[i, j] = math.atan(SAy[i,j]/SAx[i,j])
    return gA,aA




def getQabf(aA,gA,aF,gF):
    n, m = aA.shape
    GAF = np.zeros((n,m))
    AAF = np.zeros((n,m))
    QgAF = np.zeros((n,m))
    QaAF = np.zeros((n,m))
    QAF = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if(gA[i,j]>gF[i,j]):
                GAF[i,j] = gF[i,j]/gA[i,j]
            elif(gA[i,j]==gF[i,j]):
                GAF[i, j] = gF[i, j]
            else:
                GAF[i, j] = gA[i,j]/gF[i, j]
            AAF[i,j] = 1-np.abs(aA[i,j]-aF[i,j])/(math.pi/2)

            QgAF[i,j] = Tg/(1+math.exp(kg*(GAF[i,j]-Dg)))
            QaAF[i,j] = Ta/(1+math.exp(ka*(AAF[i,j]-Da)))

            QAF[i,j] = QgAF[i,j]*QaAF[i,j]

    return QAF


if __name__ == '__main__':
    path = r'D:\MMPNRMNet\image_out'
    path1 = r'D:\MMPNRMNet\data\train\GT'
    path2 = r'D:\MMPNRMNet\data1\train\GT'
    filelist = os.listdir(path)
    outputs = 0
    for file in filelist:

        strA = Image.open(os.path.join(path1,file) ).convert('L')
        strA = np.array(strA) / 255
        strB = Image.open(os.path.join(path2,file) ).convert('L')
        strB = np.array(strB) / 255
        strF = Image.open(os.path.join(path,file) ).convert('L')
        strF = np.array(strF) / 255

        gA, aA = getArray(strA)
        gB, aB = getArray(strB)
        gF, aF = getArray(strF)
        QAF = getQabf(aA, gA, aF, gF)
        QBF = getQabf(aB, gB, aF, gF)


        deno = np.sum(gA + gB)
        nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
        output = nume / deno
        print(output)
        outputs = output+outputs
    print(outputs/5)







