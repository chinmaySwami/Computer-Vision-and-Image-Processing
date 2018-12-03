# ****************** CVIP Project 3 Task 2 ***********************************
# Title          :- Image segmentation and point detection
# Author         :- Chinmay Prakash Swami
# *****************************************************************************

import cv2
import numpy as np
from functions import padd,detectPoints, generateFinalImage, performErosion
from matplotlib import pyplot as plt

image = cv2.imread("Images/point.jpg", cv2.IMREAD_GRAYSCALE)

mask = [[-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]]

maskE = [[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]]

mask = np.asarray(mask)

image = cv2.Laplacian(image, cv2.CV_32F)
imageWithPoints, sumofProductList = detectPoints(image, mask)
# imageWithPoints = abs(imageWithPoints) / max(abs(imageWithPoints))
maxSumofProduct = max(sumofProductList)
cv2.imwrite('MaskOutput.jpg', imageWithPoints)
generateFinalImage(imageWithPoints, maxSumofProduct, False, 90)

#  ************************** Task 2.2 **********************************

print("\n Task 2:- Image segmentation")
image = cv2.imread("Images/segment.jpg", cv2.IMREAD_GRAYSCALE)

uniqueVals, uniqueValsCount = np.unique(image, return_counts=True)

plt.plot(uniqueVals[1:], uniqueValsCount[1:])
plt.savefig("Histogram.png")
generateFinalImage(image, 240, True, 85)