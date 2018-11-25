# ****************** CVIP Project 3 Task 2 ***********************************
# Title          :- Image segmentation and point detection
# Author         :- Chinmay Prakash Swami
# *****************************************************************************

import numpy as np
import math,cv2

# Function padd is used to pad zeros to the image array
def padd(img):
    imgL = [[0 for imrow in range(len(img[0])+2)]for imcol in range(len(img)+2)]
    print(len(img), len(img[0]), len(imgL), len(imgL[0]))
    for imrow in range(1, len(imgL)-1):
        for imcol in range(1, len(imgL[0])-1):
            imgL[imrow][imcol] = img[imrow-1][imcol-1]
    imgL = np.asarray(imgL)
    print(type(imgL))
    return imgL

def detectPoints(image, mask):
    sumofProduct = 0
    sumofProductList = []
    rowStart = 0
    columnStart = 0
    print(image.shape, image.dtype)
    print(len(image[0]), len(image))
    image.astype(np.float64)
    imageTemp = np.zeros_like(image)
    while rowStart < 472:
        while columnStart < 353:
            imagePart = image[rowStart:rowStart+3, columnStart:columnStart+3]
            for i in range(3):
                for j in range(3):
                    sumofProduct = sumofProduct + (imagePart[i][j] * mask[i][j])
            sumofProductList.append(abs(sumofProduct))
            imageTemp[rowStart+1][columnStart+1] = abs(sumofProduct)
            columnStart += 1
            sumofProduct = 0
        rowStart += 1
        columnStart = 0
    return imageTemp, sumofProductList

def generateFinalImage(imageWithPoints, maxSumofProduct):
    pointCoordinates = []
    percent = (90/100) * maxSumofProduct
    print(maxSumofProduct, percent)
    for imrow in range(0, len(imageWithPoints)):
        for imcol in range(0, len(imageWithPoints[0])):
            if imageWithPoints[imrow][imcol] >= percent:
                imageWithPoints[imrow][imcol] = 255
                pointCoordinates.append([imrow, imcol])
            else:
                imageWithPoints[imrow][imcol] = 0
    print("Co-Ordinates of the points are:", pointCoordinates)
    cv2.imwrite('points.jpg', imageWithPoints)



