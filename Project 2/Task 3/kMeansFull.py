# ****************** CVIP Project 2 Task 3 ***********************************
# Title          :- Image Features and Homography
# Author         :- Chinmay Prakash Swami
# *****************************************************************************
import numpy as np
from functions import calculateColorDistance,findCluster,reCalculateMean
import matplotlib.pyplot  as plt
import cv2
import random

def kmeanFull(noOfClusters):
    image = cv2.imread("Images/baboon.jpg")
    imgHeight, imgWidth = image.shape[:2]
    image = np.asarray(image,dtype=np.float64) / 255
    noOfClusters = noOfClusters
    clusterCenters = []

    #  *****************   Randomly generate cluster centers  **************
    for times in range(noOfClusters):
        imageX = random.randint(0, 511)
        imageY = random.randint(0, 511)
        print(imageX, imageY)
        clusterCenters.append(image[imageX][imageY])

    # to get rid of the dtype info
    clusterCenters = np.array(clusterCenters, dtype=np.float64)
    prevClusterCenters = np.asarray(clusterCenters)
    print("cluster Centers \n", clusterCenters)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    print("image Shape ", image.shape)

    converge = False
    iteration = 0
    while (converge == False):
        print("*********** Iteration",iteration," ***********")
        iteration +=1
        # **************   Calculating euclidean distances  *************************
        colorDistance = calculateColorDistance(noOfClusters, image, clusterCenters)
        # print("colorDistance \n",colorDistance)

        # **************** Classify the colors to clusters  **************************
        ptsClassified = np.zeros((noOfClusters,image.shape[0]))
        ptsClassified = findCluster(colorDistance, ptsClassified, noOfClusters)
        print("cluster element numbers :",np.sum(ptsClassified,axis=1))

        # **************** Calculate new Cluster centers  ****************************
        prevClusterCenters = np.asarray(clusterCenters)
        clusterCenters = reCalculateMean(ptsClassified, image, colorDistance, noOfClusters, clusterCenters,prevClusterCenters)
        converge = np.array_equal(prevClusterCenters,clusterCenters)
        print("New  clusters: \n",clusterCenters)
        if iteration == 10:
            converge = True
    #
    for imgIndex in range(image.shape[0]):
        for centers in range(noOfClusters):
            if ptsClassified[centers][imgIndex] == 1:
                image[imgIndex] = clusterCenters[centers]

    image = image.reshape((imgHeight,imgWidth,3))
    image = image * 255
    image = image.astype(np.uint8)

    cv2.imwrite("task3_baboon_"+str(noOfClusters)+".jpg", image)

