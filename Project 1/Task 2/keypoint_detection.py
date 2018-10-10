# ****************** CVIP Project 1 Task 2 ***********************************
# Title          :- Keypoint Detection
# Libraries Used :- imread and imwrite of OpenCV, np for array
# Author         :- Chinmay Prakash Swami
# Used the site http://setosa.io/ev/image-kernels/ to understand how convolution works in realtime
# *****************************************************************************

import cv2
import math
import numpy as np
from functions import convolutions, findMax, findMin, plotKeyPoints,resizeImage,padd,generateGaussianFilter
from functions import sortKeypoints

# def normalizeValues(img):
#     img = (img / np.max(img))
#     img = img * 255
#     img = img.astype(np.uint8)
#
#     return img

def calculateKeypoint(dog1,dog2,dog3,octaveNo):
    stopJLoop = len(dog2[0])
    stopILoop = len(dog2)
    # Calculate terminating conditions
    if len(dog2) % 3 == 0:
        stopILoop = len(dog2)
    else:
        stopILoop = stopILoop - (len(dog2) % 3)

    if len(dog2[0]) % 3 == 0:
        stopJLoop = len(dog2[0])
    else:
        stopJLoop = stopJLoop - (len(dog2[0]) % 3)
    # start calculating keypoints
    detectedPoints = []
    for imrow in range(3, len(dog1)-3, 1):
        # if imrow == stopILoop:
        #     return detectedPoints
        for imcol in range(3, (len(dog1[0]))-3, 1):
            # if imcol == stopJLoop:
            #     imrow += 1
            #     imcol = 0
            #     continue
            mid = dog2[imrow+1][imcol+1]
            maxDOG2 = findMax(dog2, imrow, imcol)
            minDOG2 = findMin(dog2, imrow, imcol)
            if mid >= maxDOG2:
                maxDOG1 = findMax(dog1, imrow, imcol)
                maxDOG3 = findMax(dog3, imrow, imcol)
                if mid > maxDOG1 and mid > maxDOG3:
                    detectedPoints.append([(2**octaveNo) * imrow+1, (2**octaveNo) * imcol+1])

            elif mid <= minDOG2:
                minDOG1 = findMin(dog1, imrow, imcol)
                minDOG3 = findMin(dog3, imrow, imcol)
                if mid < minDOG1 and mid < minDOG3:
                    detectedPoints.append([(2 ** octaveNo) * imrow + 1, (2 ** octaveNo) * imcol + 1])

    return  detectedPoints

gaussianFilter = np.double(np.zeros((7,7)))
img = cv2.imread('task2.jpg', cv2.IMREAD_GRAYSCALE)
img = padd(img)
imgPt = img.copy()

# ********************************************** Octave 1 ******************************************************
# ************ Image 1 ************************
gaussianFilter = generateGaussianFilter(1 / math.sqrt(2), gaussianFilter)
imgo1i1 = img.copy()
print("started O1I1")
imgo1i1 = convolutions(gaussianFilter, img)
cv2.imwrite('Octave 1 Image 1.jpg', imgo1i1)

# ************ Image 2 ************************
gaussianFilter = generateGaussianFilter(1, gaussianFilter)
imgo1i2 = img.copy()
print("starting O1I2")
imgo1i2 = convolutions(gaussianFilter, img)
cv2.imwrite('Octave 1 Image 2.jpg', imgo1i2)

# ************ Image 3 ************************
gaussianFilter = generateGaussianFilter(math.sqrt(2), gaussianFilter)
imgo1i3 = img.copy()
print("starting O1I3")
imgo1i3 = convolutions(gaussianFilter, img)
cv2.imwrite('Octave 1 Image 3.jpg', imgo1i3)

# ************ Image 4 ************************
gaussianFilter = generateGaussianFilter(2, gaussianFilter)
imgo1i4 = img.copy()
print("starting O1I4")
imgo1i4 = convolutions(gaussianFilter, img)
cv2.imwrite('Octave 1 Image 4.jpg', imgo1i4)

# ************ Image 5 ************************
gaussianFilter = generateGaussianFilter(2 * math.sqrt(2), gaussianFilter)
imgo1i5 = img.copy()
print("starting O1I5")
imgo1i5 = convolutions(gaussianFilter, img)
cv2.imwrite('Octave 1 Image 5.jpg', imgo1i5)

# ***************** Calculate DOG *************************
print("calculating DOG for octave 1")
DOGO11 = imgo1i1 - imgo1i2
DOGO12 = imgo1i2 - imgo1i3
DOGO13 = imgo1i3 - imgo1i4
DOGO14 = imgo1i4 - imgo1i5

# ***************** Calculate Keypoints *************************
print(len(DOGO11), len(DOGO11[0]))
print("calculating Keypoints")
keypointsToPlot = []
keypointsToPlot = keypointsToPlot + calculateKeypoint(DOGO11, DOGO12, DOGO13,0)
keypointsToPlot = keypointsToPlot + calculateKeypoint(DOGO12, DOGO13, DOGO14,0)
print(keypointsToPlot)
print(len(keypointsToPlot))

# ************************************************** Octave 2 *******************************************************
imgo2 = resizeImage(img)
cv2.imwrite("task2O2.jpg", imgo2)

# ************ Image 1 ************************
gaussianFilter = generateGaussianFilter(math.sqrt(2), gaussianFilter)
imgo2i1 = imgo2.copy()
print("starting O2I1")
imgo2i1 = convolutions(gaussianFilter, imgo2)
cv2.imwrite('Octave 2 Image 1.jpg', imgo2i1)

# ************ Image 2 ************************
gaussianFilter = generateGaussianFilter(2, gaussianFilter)
imgo2i2 = imgo2.copy()
print("starting O2I2")
imgo2i2 = convolutions(gaussianFilter, imgo2)
cv2.imwrite('Octave 2 Image 2.jpg', imgo2i2)

# ************ Image 3 ************************
gaussianFilter = generateGaussianFilter(2 * math.sqrt(2), gaussianFilter)
imgo2i3 = imgo2.copy()
print("starting O2I3")
imgo2i3 = convolutions(gaussianFilter, imgo2)
cv2.imwrite('Octave 2 Image 3.jpg', imgo2i3)

# ************ Image 4 ************************
gaussianFilter = generateGaussianFilter(4, gaussianFilter)
imgo2i4 = imgo2.copy()
print("starting O2I4")
imgo2i4 = convolutions(gaussianFilter, imgo2)
cv2.imwrite('Octave 2 Image 4.jpg', imgo2i4)

# ************ Image 5 ************************
gaussianFilter = generateGaussianFilter(4 * math.sqrt(2), gaussianFilter)
imgo2i5 = imgo2.copy()
print("starting O2I5")
imgo2i5 = convolutions(gaussianFilter, imgo2)
cv2.imwrite('Octave 2 Image 5.jpg', imgo2i5)

# ***************** Calculate DOG *************************
print("calculating DOG for octave 2")
DOGO21 = imgo2i1 - imgo2i2
DOGO22 = imgo2i2 - imgo2i3
DOGO23 = imgo2i3 - imgo2i4
DOGO24 = imgo2i4 - imgo2i5
# ***************** Calculate Keypoints *************************
print("calculating Keypoints")
# keypointsToPlot = []
keypointsToPlot = keypointsToPlot + calculateKeypoint(DOGO21, DOGO22, DOGO23, 1)
keypointsToPlot = keypointsToPlot + calculateKeypoint(DOGO22, DOGO23, DOGO24, 1)
print(keypointsToPlot)
print(len(keypointsToPlot))
print(len(keypointsToPlot))

cv2.imshow('DOGO21', DOGO21)   # Displays the image from img_opy object[vertical edges]
cv2.waitKey(0)            # waits for user to press any key
cv2.destroyAllWindows()

cv2.imshow('DOGO22', DOGO22)   # Displays the image from img_opy object[vertical edges]
cv2.waitKey(0)            # waits for user to press any key
cv2.destroyAllWindows()

cv2.imshow('DOGO23', DOGO23)   # Displays the image from img_opy object[vertical edges]
cv2.waitKey(0)            # waits for user to press any key
cv2.destroyAllWindows()

cv2.imshow('DOGO24', DOGO24)   # Displays the image from img_opy object[vertical edges]
cv2.waitKey(0)            # waits for user to press any key
cv2.destroyAllWindows()


# ************************************************** Octave 3 *******************************************************
imgo3 = resizeImage(imgo2)
cv2.imwrite("task2O3.jpg", imgo3)

# ************ Image 1 ************************
gaussianFilter = generateGaussianFilter(2 * math.sqrt(2), gaussianFilter)
imgo3i1 = imgo3.copy()
print("starting O3I1")
imgo3i1 = convolutions(gaussianFilter, imgo3)
cv2.imwrite('Octave 3 Image 1.jpg', imgo3i1)

# ************ Image 2 ************************
gaussianFilter = generateGaussianFilter(4, gaussianFilter)
imgo3i2 = imgo3.copy()
print("starting O3I2")
imgo3i2 = convolutions(gaussianFilter, imgo3)
cv2.imwrite('Octave 3 Image 2.jpg', imgo3i2)

# ************ Image 3 ************************
gaussianFilter = generateGaussianFilter(4 * math.sqrt(2), gaussianFilter)
imgo3i3 = imgo3.copy()
print("starting O3I3")
imgo3i3 = convolutions(gaussianFilter, imgo3)
cv2.imwrite('Octave 3 Image 3.jpg', imgo3i3)

# ************ Image 4 ************************
gaussianFilter = generateGaussianFilter(8, gaussianFilter)
imgo3i4 = imgo3.copy()
print("starting O3I4")
imgo3i4 = convolutions(gaussianFilter, imgo3)
cv2.imwrite('Octave 3 Image 4.jpg', imgo3i4)

# ************ Image 5 ************************
gaussianFilter = generateGaussianFilter(8 * math.sqrt(2), gaussianFilter)
imgo3i5 = imgo3.copy()
print("starting O3I5")
imgo3i5 = convolutions(gaussianFilter, imgo3)
cv2.imwrite('Octave 3 Image 5.jpg', imgo3i5)

# ***************** Calculate DOG *************************
print("calculating DOG for octave 3")
DOGO31 = imgo3i1 - imgo3i2
DOGO32 = imgo3i2 - imgo3i3
DOGO33 = imgo3i3 - imgo3i4
DOGO34 = imgo3i4 - imgo3i5

cv2.imshow('DOGO31', DOGO31)   # Displays the image from img_opy object[vertical edges]
cv2.waitKey(0)            # waits for user to press any key
cv2.destroyAllWindows()

cv2.imshow('DOGO32', DOGO32)   # Displays the image from img_opy object[vertical edges]
cv2.waitKey(0)            # waits for user to press any key
cv2.destroyAllWindows()

cv2.imshow('DOGO33', DOGO33)   # Displays the image from img_opy object[vertical edges]
cv2.waitKey(0)            # waits for user to press any key
cv2.destroyAllWindows()

cv2.imshow('DOGO34', DOGO34)   # Displays the image from img_opy object[vertical edges]
cv2.waitKey(0)            # waits for user to press any key
cv2.destroyAllWindows()


print("calculating Keypoints")
# keypointsToPlot = []
keypointsToPlot = keypointsToPlot + calculateKeypoint(DOGO31, DOGO32, DOGO33, 2)
keypointsToPlot = keypointsToPlot + calculateKeypoint(DOGO32, DOGO33, DOGO34, 2)
print(keypointsToPlot)
print(len(keypointsToPlot))
print(len(keypointsToPlot))

# ************************************************** Octave 4 *******************************************************
imgo4 = resizeImage(imgo3)
cv2.imwrite("task2O4.jpg", imgo4)

# ************ Image 1 ************************
gaussianFilter = generateGaussianFilter(4 * math.sqrt(2), gaussianFilter)
imgo4i1 = imgo4.copy()
print("starting O4I1")
imgo4i1 = convolutions(gaussianFilter, imgo4)
cv2.imwrite('Octave 4 Image 1.jpg', imgo4i1)

# ************ Image 2 ************************
gaussianFilter = generateGaussianFilter(8, gaussianFilter)
imgo4i2 = imgo4.copy()
print("starting O4I2")
imgo4i2 = convolutions(gaussianFilter, imgo4)
cv2.imwrite('Octave 4 Image 2.jpg', imgo4i2)

# ************ Image 3 ************************
gaussianFilter = generateGaussianFilter(8 * math.sqrt(2), gaussianFilter)
imgo4i3 = imgo4.copy()
print("starting O4I3")
imgo4i3 = convolutions(gaussianFilter, imgo4)
cv2.imwrite('Octave 4 Image 3.jpg', imgo4i3)

# ************ Image 4 ************************
gaussianFilter = generateGaussianFilter(16, gaussianFilter)
imgo4i4 = imgo4.copy()
print("starting O4I4")
imgo4i4 = convolutions(gaussianFilter, imgo4)
cv2.imwrite('Octave 4 Image 4.jpg', imgo4i4)

# ************ Image 5 ************************
gaussianFilter = generateGaussianFilter(16 * math.sqrt(2), gaussianFilter)
imgo4i5 = imgo4.copy()
print("starting O4I5")
imgo4i5 = convolutions(gaussianFilter, imgo4)
cv2.imwrite('Octave 4 Image 5.jpg', imgo4i5)

# ***************** Calculate DOG *************************
print("calculating DOG for octave 3")
DOGO41 = imgo4i1 - imgo4i2
DOGO42 = imgo4i2 - imgo4i3
DOGO43 = imgo4i3 - imgo4i4
DOGO44 = imgo4i4 - imgo4i5

print("calculating Keypoints")
# keypointsToPlot = []
keypointsToPlot = keypointsToPlot + calculateKeypoint(DOGO41, DOGO42, DOGO43, 3)
keypointsToPlot = keypointsToPlot + calculateKeypoint(DOGO42, DOGO43, DOGO44, 3)
print(keypointsToPlot)
print(len(keypointsToPlot))
print(len(keypointsToPlot))
# ***************** Calculating Keypoints ******************
imgPt = plotKeyPoints(imgPt, keypointsToPlot)
cv2.imwrite("task2_Keypoints.jpg", imgPt)

# ***************** Calculating top 5 keypoints ******************
keypointsToPlot = sortKeypoints(keypointsToPlot)
print("Top 5 Keypoints are ")
for i in range(0,5):
    print(keypointsToPlot[i][0],keypointsToPlot[i][1])

imgPt2 = img.copy()
imgPt2 = plotKeyPoints(imgPt2, [[4,4],[13,27],[17,25],[13,35],[23,30]])
cv2.imwrite("task2_Keypoints_top5.jpg", imgPt2)
#
# cv2.imshow("DOGO21", DOGO21)
# cv2.waitKey(0)            # waits for user to press any key
# cv2.destroyAllWindows()   # closes all output windows when a key is pressed
# cv2.imshow("DOGO22", DOGO22)
# cv2.imshow("DOGO23", DOGO23)
# cv2.imshow("DOGO24", DOGO24)
