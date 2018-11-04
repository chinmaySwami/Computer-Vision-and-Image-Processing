import numpy as np
from functions import calculateColorDistance,findCluster,reCalculateMean
import matplotlib.pyplot  as plt
import cv2
import random

image = cv2.imread("Images/baboon.jpg")
imgHeight, imgWidth = image.shape[:2]
image = np.asarray(image,dtype=np.float64) / 255
noOfClusters = 3
clusterCenters = []

#  *****************   Randomly generate cluster centers  **************
for times in range(noOfClusters):
    imageX = random.randint(0, 511)
    imageY = random.randint(0, 511)
    print(imageX, imageY)
    clusterCenters.append(image[imageX][imageY])

# to get rid of the dtype info
clusterCenters = np.array(clusterCenters, dtype=np.float64) / 255
print("cluster Centers \n", clusterCenters)
image = image.reshape((image.shape[0] * image.shape[1], 3))
print("image Shape ", image.shape)

for i in range(3):

    # **************   Calculating euclidean distances  *************************
    colorDistance = calculateColorDistance(noOfClusters, image, clusterCenters)
    # print("colorDistance \n",colorDistance)

    # **************** Classify the colors to clusters  **************************
    ptsClassified = np.zeros((noOfClusters, image.shape[0]))
    ptsClassified = findCluster(colorDistance, ptsClassified, noOfClusters)
    # print("classification vector: \n",ptsClassified)

    # **************** Calculate new Cluster centers  ****************************
    clusterCenters = reCalculateMean(ptsClassified, colorDistance, noOfClusters, clusterCenters)
    clusterCenters = np.asarray(clusterCenters)
    print("New  clusters: \n",clusterCenters)
#
for centers in range(noOfClusters):
    for imgIndex in range(image.shape[0]):
        if ptsClassified[centers][imgIndex] == 1:
            image[imgIndex] = clusterCenters[centers]

image = image.reshape((imgHeight,imgWidth,3))
colorDistance = colorDistance.reshape((imgHeight,imgWidth,noOfClusters))
image = image * 255
image = image.astype(np.uint8)

colorDistance = colorDistance * 255
colorDistance = colorDistance.astype(np.uint8)

print(colorDistance)
cv2.imwrite("task3_baboon_3.jpg", image)

