# ****************** CVIP Project 3 Task 1 ***********************************
# Title          :- Morphology image processing
# Author         :- Chinmay Prakash Swami
# *****************************************************************************
import cv2
import numpy as np
from functions import padd, performErosion, performDilation,showImage

image = cv2.imread('Images/noise.jpg', cv2.IMREAD_GRAYSCALE)
print(image.dtype)
image = padd(image)
print(image.dtype)
image = image / 255
image = np.asarray(image, dtype=np.int32)

structuringElement = [[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]]

structuringElement = np.asarray(structuringElement, dtype=np.int32)

#  Method 1 Opening and Closing :- Erosion -> Dilation -> Dilation -> Erosion
#  Perform Erosion
imageTemp = performErosion(image, structuringElement)

# Perform Dilation
imageTemp = performDilation(imageTemp, structuringElement)

# Perform Dilation
imageTemp = performDilation(imageTemp, structuringElement)

#  Perform Erosion
imageMethod1 = performErosion(imageTemp, structuringElement)
imageMethod1 = imageTemp * 255
cv2.imwrite("res_noise1.jpg", imageMethod1)

#  Method 2  Closing and Opening :- Dilation -> Erosion -> Erosion -> Dilation
image = cv2.imread('Images/noise.jpg', cv2.IMREAD_GRAYSCALE)
print(image.dtype)
image = padd(image)
print(image.dtype)
image = image / 255
image = np.asarray(image, dtype=np.int32)

# Perform Dilation
imageTemp = performDilation(image, structuringElement)

#  Perform Erosion
imageTemp = performErosion(imageTemp, structuringElement)

#  Perform Erosion
imageTemp = performErosion(imageTemp, structuringElement)

# Perform Dilation
imageMethod2 = performDilation(imageTemp, structuringElement)

imageMethod2 = imageTemp * 255
cv2.imwrite("res_noise2.jpg", imageMethod2)

