# ****************** CVIP Project 3 Task 2 ***********************************
# Title          :- Image segmentation and point detection
# Author         :- Chinmay Prakash Swami
# *****************************************************************************

import cv2
import numpy as np
from functions import padd,detectPoints, generateFinalImage
from matplotlib import pyplot as plt

image = cv2.imread("Images/point.jpg", cv2.IMREAD_GRAYSCALE)
# image = cv2.cvtColor(imageC, cv2.COLOR_BGR2GRAY)

# image = padd(image)
# plt.hist(image.ravel(),256,[0,256])
# plt.show()

cv2.imwrite('paddedIMage.jpg', image)
print(image.dtype)

mask = [[-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]]

mask = np.asarray(mask)
imageWithPoints, sumofProductList = detectPoints(image, mask)
# imageWithPoints = abs(imageWithPoints) / max(abs(imageWithPoints))
maxSumofProduct = max(sumofProductList)

cv2.imwrite('MaskOutput.jpg', imageWithPoints)
generateFinalImage(imageWithPoints, maxSumofProduct)
