# ****************** CVIP Project 2 Task 3 ***********************************
# Title          :- K-Means Clustering
# Author         :- Chinmay Prakash Swami
# *****************************************************************************
import numpy as np
from functions import calculateDistance,chooseCenteroid
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

#  Extract X & Y co-ordinates
arrayX = np.asarray(X)
arrayY = np.asarray(X)
xX = arrayX[: , :1].tolist()
yX = arrayY[: , 1:].tolist()

plt.scatter(xX,yX,facecolors='none',edgecolor = XClassified,marker="^")
for i in range(len(X)):
    plt.text(str(xX[i]),str(yX[i]),s='1')
plt.show()


# for x,y,color in zip(xX,yX,XClassified):
#     plt.scatter(x,y,color = color)



