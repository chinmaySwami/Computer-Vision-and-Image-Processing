import numpy as np
from functions import calculateColorDistance
import matplotlib.pyplot  as plt
import cv2
import random

image = cv2.imread("Images/baboon.jpg")

noOfClusters = 3

clusterCenters = []

for times in range(noOfClusters):
    imageX = random.randint(0, 511)
    imageY = random.randint(0, 511)
    print(imageX,imageY)
    clusterCenters.append(image[imageX][imageY])
# to get rid of the dtype info
clusterCenters = np.array(clusterCenters)
print(clusterCenters)
image = image.reshape((image.shape[0] * image.shape[1], 3))
print(image.shape)
colorDistance = calculateColorDistance(clusterCenters, image, clusterCenters)
