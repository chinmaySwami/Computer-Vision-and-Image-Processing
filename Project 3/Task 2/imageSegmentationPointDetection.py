# ****************** CVIP Project 3 Task 2 ***********************************
# Title          :- Image segmentation and point detection
# Author         :- Chinmay Prakash Swami
# *****************************************************************************

import cv2
import numpy as np
from functions import padd,detectPoints, generateFinalImage, performErosion
from matplotlib import pyplot as plt

image = cv2.imread("Images/point.jpg", cv2.IMREAD_GRAYSCALE)
# image = cv2.cvtColor(imageC, cv2.COLOR_BGR2GRAY)

# image = padd(image)
print(image.dtype)

mask = [[-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]]

maskE = [[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]]

mask = np.asarray(mask)

image = cv2.Laplacian(image,cv2.CV_32F)
imageWithPoints, sumofProductList = detectPoints(image, mask)
# imageWithPoints = abs(imageWithPoints) / max(abs(imageWithPoints))
maxSumofProduct = max(sumofProductList)

cv2.imwrite('MaskOutput.jpg', imageWithPoints)
generateFinalImage(imageWithPoints, maxSumofProduct, False, 90)

#  ************************** Task 2.2 **********************************

image = cv2.imread("Images/segment.jpg", cv2.IMREAD_GRAYSCALE)

uniqueVals, uniqueValsCount = np.unique(image, return_counts=True)

# plt.hist(image.ravel(),256,[1,256])
# plt.show()

plt.plot(uniqueVals[1:], uniqueValsCount[1:])
plt.show()

print(uniqueVals)
print(uniqueValsCount)

generateFinalImage(image, 240, True, 83)