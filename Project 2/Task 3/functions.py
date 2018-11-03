
import numpy as np
import math

def calculateDistance(clusterCenters,X,XDistance):
    for dataPts in range(len(X)):
        XDistanceTemp = []
        for centers in range(len(clusterCenters)):
            # Center points are X0,Y0 and Datapoints are X1,Y1
            d1 = ((clusterCenters[centers][0] - X[dataPts][0])**2)
            d2 = ((clusterCenters[centers][1] - X[dataPts][1])**2)
            distance = (d1 + d2)**0.5
            XDistanceTemp.append(distance)
        XDistance.append(XDistanceTemp)
    return XDistance

def chooseCenteroid(XDistance,XClassified):
    for distances in range(len(XDistance)):
        print(XDistance[distances])
        minValIndexLoc = XDistance[distances].index(min(XDistance[distances]))
        print(minValIndexLoc)
        if minValIndexLoc == 0:  # belongs to cluster 1
            XClassified.append('Red')
        elif minValIndexLoc == 1:  # belongs to cluster 2
            XClassified.append('Blue')
        elif minValIndexLoc == 2:  # belongs to cluster 3
            XClassified.append('Green')
    return XClassified


