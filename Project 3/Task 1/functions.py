# ****************** CVIP Project 3 Task 1 ***********************************
# Title          :- Morphology image processing
# Author         :- Chinmay Prakash Swami
# *****************************************************************************

import numpy as np
import math,cv2

def showImage(image):
    cv2.imshow('X_direction', image)  # Displays the image from img_opy object[vertical edges]
    cv2.waitKey(0)  # waits for user to press any key
    cv2.destroyAllWindows()  # closes all output windows when a key is pressed

# Function padd is used to pad zeros to the image array
def padd(img):
    imgL = [[0 for imrow in range(len(img[0])+2)]for imcol in range(len(img)+2)]
    print(len(img), len(img[0]), len(imgL), len(imgL[0]))
    for imrow in range(1, len(imgL)-1):
        for imcol in range(1, len(imgL[0])-1):
            imgL[imrow][imcol] = img[imrow-1][imcol-1]
    imgL = np.asarray(imgL, dtype=np.uint8)
    print(type(imgL))
    return imgL

def checkErosionCondition(imagePart, mask):
    for i in range(3):
        for j in range(3):
            if imagePart[i][j] != mask[i][j]:
                return False
    return True

def checkDilationCondition(imagePart, mask):
    for i in range(3):
        for j in range(3):
            if imagePart[i][j] == mask[i][j]:
                return True
    return False

def performErosion(image, mask):
    print("Performing Erosion: \n")
    rowStart = 0
    columnStart = 0
    allOnes = True
    imageTemp = np.zeros_like(image)
    while rowStart < 307:
        while columnStart < 348:
            imagePart = image[rowStart:rowStart+3, columnStart:columnStart+3]
            allOnes = checkErosionCondition(imagePart, mask)
            if allOnes:
                imageTemp[rowStart+1][columnStart+1] = 1
            columnStart += 1
        rowStart += 1
        columnStart = 0
        allOnes = True
    return imageTemp

def performDilation(image, mask):
    print("Performing Dilation: \n")
    rowStart = 0
    columnStart = 0
    anyOnes = False
    imageTemp = image.copy()
    while rowStart < 307:
        while columnStart < 348:
            imagePart = image[rowStart:rowStart+3, columnStart:columnStart+3]
            anyOnes = checkDilationCondition(imagePart, mask)
            if anyOnes:
                imageTemp[rowStart+1][columnStart+1] = 1
            columnStart += 1
        rowStart += 1
        columnStart = 0
        anyOnes = False
    return imageTemp




