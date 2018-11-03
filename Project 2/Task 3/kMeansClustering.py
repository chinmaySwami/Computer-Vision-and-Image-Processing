# ****************** CVIP Project 2 Task 3 ***********************************
# Title          :- K-Means Clustering
# Author         :- Chinmay Prakash Swami
# *****************************************************************************
import numpy as np
from functions import calculateDistance,chooseCenteroid,caculateNewMean,plotClusters
import matplotlib.pyplot  as plt

clusterCenters = [[6.2, 3.2],
                  [6.6, 3.7],
                  [6.5, 3.0]]

X = [[5.9, 3.2],
     [4.6, 2.9],
     [6.2, 2.8],
     [4.7, 3.2],
     [5.5, 4.2],
     [5.0, 3.0],
     [4.9, 3.1],
     [6.7, 3.1],
     [5.1, 3.8],
     [6.0, 3.0]]

XDistance = []
# Calculate the distances of points from clusters
XDistance = calculateDistance(clusterCenters, X, XDistance)
XClassified = []
print(len(XDistance), XDistance)
#  Classify the points based on distance calculated earlier
XClassified = chooseCenteroid(XDistance,XClassified)
print(XClassified)

# Plot the cluesters
plotClusters(X,clusterCenters,XClassified)

# Task 3.2:- Calculate new cluster centroids

clusterCenters = caculateNewMean(X,XClassified)
print(clusterCenters)

plotClusters(X,clusterCenters,XClassified)

# Task 3.3 :- calculate euclidian disctances for new mean and generate clusters

XDistance = []
# Calculate the distances of points from clusters
XDistance = calculateDistance(clusterCenters, X, XDistance)
XClassified = []
print(len(XDistance), XDistance)
#  Classify the points based on distance calculated earlier
XClassified = chooseCenteroid(XDistance,XClassified)
print(XClassified)

plotClusters(X,clusterCenters,XClassified)

