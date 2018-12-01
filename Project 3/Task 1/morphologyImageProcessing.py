# ****************** CVIP Project 3 Task 1 ***********************************
# Title          :- Morphology image processing
# Author         :- Chinmay Prakash Swami
# *****************************************************************************
import cv2
import numpy as np
from functions import padd, performErosion, performDilation,showImage

imageA1 = cv2.imread('Images/noise.jpg', cv2.IMREAD_GRAYSCALE)
# imageA1 = padd(imageA1)
imageA1 = imageA1 / 255
imageA1 = np.asarray(imageA1, dtype=np.int32)

structuringElement = [[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]]

structuringElement = np.asarray(structuringElement, dtype=np.int32)

#  Method 1 Opening and Closing :- Erosion -> Dilation -> Dilation -> Erosion
#  Perform Erosion
imageA1E = performErosion(imageA1, structuringElement)

# Perform Dilation
imageA1ED = performDilation(imageA1E, structuringElement)

# Perform Dilation
imageA1EDD = performDilation(imageA1ED, structuringElement)

#  Perform Erosion
imageMethod1 = performErosion(imageA1EDD, structuringElement)
imageMethod1B = imageMethod1
imageMethod1 = imageMethod1 * 255
cv2.imwrite("res_noise1.jpg", imageMethod1)

#  Method 2  Closing and Opening :- Dilation -> Erosion -> Erosion -> Dilation
imageA2 = cv2.imread('Images/noise.jpg', cv2.IMREAD_GRAYSCALE)
print(imageA2.dtype)
# imageA2 = padd(imageA2)
print(imageA2.dtype)
imageA2 = imageA2 / 255
imageA2 = np.asarray(imageA2, dtype=np.int32)

# Perform Dilation
imageA2D = performDilation(imageA2, structuringElement)

#  Perform Erosion
imageA2DE = performErosion(imageA2D, structuringElement)

#  Perform Erosion
imageA2DEE = performErosion(imageA2DE, structuringElement)

# Perform Dilation
imageMethod2 = performDilation(imageA2DEE, structuringElement)
imageMethod2B = imageMethod2
imageMethod2 = imageMethod2 * 255
cv2.imwrite("res_noise2.jpg", imageMethod2)

# ****************Extraction of boundaries Res_1 ********************

imageR1E = performErosion(imageMethod1B, structuringElement)

# imageR1D = performDilation(imageR1, structuringElement)
# imageR1D = performDilation(imageR1D, structuringElement)

imageR1B = imageMethod1B - imageR1E
imageR1B = imageR1B * 255
cv2.imwrite("res_bound1.jpg", imageR1B)

# Extraction of boundaries Res_2

imageR2E = performErosion(imageMethod2B, structuringElement)

# imageR2D = performDilation(imageR2D, structuringElement)

imageR2B = imageMethod2B - imageR2E
imageR2B = imageR2B * 255
cv2.imwrite("res_bound2.jpg", imageR2B)