import cv2
import math
import numpy as np

# Function padd is used to pad zeros to the image array
def padd(img):
# uint8 function of np is used for type castig purpose
    imgL = np.float32(np.zeros((len(img)+6, len(img[0])+6)))
    print(len(img), len(img[0]), len(imgL), len(imgL[0]))

    for imrow in range(3, len(imgL)-3):
        for imcol in range(3, len(imgL[0])-3):
            imgL[imrow][imcol] = img[imrow-3][imcol-3]

    return imgL

def convolutions(gaussianFilter,img):
    imgCp = img.copy()
    for imrow in range(0, len(img)):
        for imcol in range(0, len(img[0])):
            pixvalx = 0
            if imrow > 2 and imcol > 2 and imrow < len(img) - 3 and imcol < len(img[0]) - 3:
                pixvalx = ((gaussianFilter[0][0] * img[imrow - 3][imcol - 3]) +
                           (gaussianFilter[0][1] * img[imrow - 3][imcol - 2]) +
                           (gaussianFilter[0][2] * img[imrow - 3][imcol - 1]) +
                           (gaussianFilter[0][3] * img[imrow - 3][imcol]) +
                           (gaussianFilter[0][4] * img[imrow - 3][imcol + 1]) +
                           (gaussianFilter[0][5] * img[imrow - 3][imcol + 2]) +
                           (gaussianFilter[0][6] * img[imrow - 3][imcol + 3]) +
                           (gaussianFilter[1][0] * img[imrow - 2][imcol - 3]) +
                           (gaussianFilter[1][1] * img[imrow - 2][imcol - 2]) +
                           (gaussianFilter[1][2] * img[imrow - 2][imcol - 1]) +
                           (gaussianFilter[1][3] * img[imrow - 2][imcol]) +
                           (gaussianFilter[1][4] * img[imrow - 2][imcol + 1]) +
                           (gaussianFilter[1][5] * img[imrow - 2][imcol + 2]) +
                           (gaussianFilter[1][6] * img[imrow - 2][imcol + 3]) +
                           (gaussianFilter[2][0] * img[imrow - 1][imcol - 3]) +
                           (gaussianFilter[2][1] * img[imrow - 1][imcol - 2]) +
                           (gaussianFilter[2][2] * img[imrow - 1][imcol - 1]) +
                           (gaussianFilter[2][3] * img[imrow - 1][imcol]) +
                           (gaussianFilter[2][4] * img[imrow - 1][imcol + 1]) +
                           (gaussianFilter[2][5] * img[imrow - 1][imcol + 2]) +
                           (gaussianFilter[2][6] * img[imrow - 1][imcol + 3]) +
                           (gaussianFilter[3][0] * img[imrow][imcol - 3]) +
                           (gaussianFilter[3][1] * img[imrow][imcol - 2]) +
                           (gaussianFilter[3][2] * img[imrow][imcol - 1]) +
                           (gaussianFilter[3][3] * img[imrow][imcol]) +
                           (gaussianFilter[3][4] * img[imrow][imcol + 1]) +
                           (gaussianFilter[3][5] * img[imrow][imcol + 2]) +
                           (gaussianFilter[3][6] * img[imrow][imcol + 3]) +
                           (gaussianFilter[4][0] * img[imrow + 1][imcol - 3]) +
                           (gaussianFilter[4][1] * img[imrow + 1][imcol - 2]) +
                           (gaussianFilter[4][2] * img[imrow + 1][imcol - 1]) +
                           (gaussianFilter[4][3] * img[imrow + 1][imcol]) +
                           (gaussianFilter[4][4] * img[imrow + 1][imcol + 1]) +
                           (gaussianFilter[4][5] * img[imrow + 1][imcol + 2]) +
                           (gaussianFilter[4][6] * img[imrow + 1][imcol + 3]) +
                           (gaussianFilter[5][0] * img[imrow + 2][imcol - 3]) +
                           (gaussianFilter[5][1] * img[imrow + 2][imcol - 2]) +
                           (gaussianFilter[5][2] * img[imrow + 2][imcol - 1]) +
                           (gaussianFilter[5][3] * img[imrow + 2][imcol]) +
                           (gaussianFilter[5][4] * img[imrow + 2][imcol + 1]) +
                           (gaussianFilter[5][5] * img[imrow + 2][imcol + 2]) +
                           (gaussianFilter[5][6] * img[imrow + 2][imcol + 3]) +
                           (gaussianFilter[6][0] * img[imrow + 3][imcol - 3]) +
                           (gaussianFilter[6][1] * img[imrow + 3][imcol - 2]) +
                           (gaussianFilter[6][2] * img[imrow + 3][imcol - 1]) +
                           (gaussianFilter[6][3] * img[imrow + 3][imcol]) +
                           (gaussianFilter[6][4] * img[imrow + 3][imcol + 1]) +
                           (gaussianFilter[6][5] * img[imrow + 3][imcol + 2]) +
                           (gaussianFilter[6][6] * img[imrow + 3][imcol + 3]))

            imgCp[imrow][imcol] = pixvalx  # calculating for horizontal edges
#   imgCp = (imgCp / np.max(imgCp)) # using this shows correct image in imshow but gives black image in inwrite
    return imgCp

# Function to create a gaussian filter
def generateGaussianFilter(sigma,gaussianFilter):
    sigmaVal = 2 * sigma * sigma
    for x in range(0, 7):
        for y in range(0, 7):
            gaussianFilter[x][y] = (math.exp(-((((y-3)**2) + ((3-x)**2)) / sigmaVal))) / (math.pi * sigmaVal)
    gaussianFilterSum = gaussianFilter.sum()
    for x in range(0, 6):
        for y in range(0, 6):
            gaussianFilter[x][y] = gaussianFilter[x][y] / gaussianFilterSum

    return gaussianFilter

# Function to find maximum value in array
def findMax(img,startIndexI,startIndexJ):
    tempMax = img[startIndexI][startIndexJ]
    for imrow in range(startIndexI, startIndexI+3):
        for imcol in range(startIndexJ, startIndexJ+3):
            if img[imrow][imcol] > tempMax:
                tempMax = img[imrow][imcol]
    return tempMax

# Function to find minimum value in array
def findMin(img,startIndexI,startIndexJ):
    tempMin = img[startIndexI][startIndexJ]
    for imrow in range(startIndexI, startIndexI+3):
        for imcol in range(startIndexJ, startIndexJ+3):
            if img[imrow][imcol] < tempMin:
                tempMin = img[imrow][imcol]
    return tempMin

def plotKeyPoints(img,keypointList):
    for i in range(0, len(keypointList)):
        x = keypointList[i][0]
        y = keypointList[i][1]
        img[x][y] = 255
    return  img

def resizeImage(img):
    imgCp = img.copy()
    # for imRow in range(0, len(img)):
    #     if imRow % 2 != 0:
    #         imgCp = np.delete(imgCp, imRow, axis=1)
    imgCp = imgCp[::2, ::2]
    return imgCp

def calculateEuclideanDistance(keypointList):
    eucDistance = []
    for i in range(0, len(keypointList)):
        eucDistance.append((((0-keypointList[i][0])**2) + ((0-keypointList[i][1])**2 ))**0.5)
    return eucDistance

def sortKeypoints(keypointList):
    print("Started Sorting")
    eucDistance = calculateEuclideanDistance(keypointList)
    sortedkeypointList = [val for _,val in sorted(zip(eucDistance,keypointList))]
    return  sortedkeypointList

# # Testing logic for slicing to do resize
# a = np.reshape(np.arange (40),(5,8))
# print(a)
# print(a[::2,::2])
