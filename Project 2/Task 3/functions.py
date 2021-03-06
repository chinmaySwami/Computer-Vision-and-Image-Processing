import numpy as np
import matplotlib.pyplot as plt
import math

clusterCentersColor = ['Red', 'Green', 'Blue']

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
        minValIndexLoc = XDistance[distances].index(min(XDistance[distances]))
        if minValIndexLoc == 0:  # belongs to cluster 1
            XClassified.append('Red')
        elif minValIndexLoc == 1:  # belongs to cluster 2
            XClassified.append('Green')
        elif minValIndexLoc == 2:  # belongs to cluster 3
            XClassified.append('Blue')
    return XClassified

def caculateNewMean(X,XClassified):
    sumRedX = 0
    sumRedY = 0
    countRed = 0
    sumBlueX = 0
    sumBlueY = 0
    countBlue = 0
    sumGreenX = 0
    sumGreenY = 0
    countGreen = 0

    for index in range(len(X)):
        if XClassified[index] == 'Red':  # belongs to cluster 1
            sumRedX += X[index][0]
            sumRedY += X[index][1]
            countRed += 1
        elif XClassified[index] == 'Blue':  # belongs to cluster 2
            sumBlueX += X[index][0]
            sumBlueY += X[index][1]
            countBlue += 1
        elif XClassified[index] == 'Green':  # belongs to cluster 3
            sumGreenX += X[index][0]
            sumGreenY += X[index][1]
            countGreen += 1
    newCenters = [[sumRedX/countRed, sumRedY/countRed],
                  [sumGreenX / countGreen, sumGreenY / countGreen],
                  [sumBlueX / countBlue, sumBlueY / countBlue]]

    return newCenters

def plotClusters(X,clusterCenters,XClassified,fineName):
    #  Extract X & Y co-ordinates
    arrayX = np.asarray(X)
    arrayClusterCenters = np.asarray(clusterCenters)
    xX = arrayX[:, :1].tolist()
    yX = arrayX[:, 1:].tolist()
    centerx = arrayClusterCenters[:, :1].tolist()
    centery = arrayClusterCenters[:, 1:].tolist()

    plt.scatter(centerx, centery, color=clusterCentersColor, marker="o")

    for i in range(len(clusterCenters)):
        plt.text(centerx[i][0] + 0.005, centery[i][0], s='(' + str(centerx[i][0]) + ',' + str(centery[i][0]) + ')')

    for i in range(len(X)):
        plt.scatter(xX[i], yX[i], color=XClassified[i], marker="^")
        plt.text(xX[i][0] + 0.005, yX[i][0], s='(' + str(xX[i][0]) + ',' + str(yX[i][0]) + ')')
    # plt.show()
    plt.savefig(fineName)
    plt.clf()

def calculateColorDistance(noOfClusters, image, clusterCenters):
    colorDistance = np.zeros((noOfClusters, image.shape[0]))
    print("in function")
    print(image.shape[0])
    for imgX in range(image.shape[0]):
        for centers in range(noOfClusters):
            x = image[imgX]
            y = clusterCenters[centers]
            distance = np.linalg.norm(x-y)
            colorDistance[centers][imgX] = distance
    return colorDistance

def findCluster(colorDistance, ptsClassified, noOfClusters):
    colorDistancecolumns = []
    for imgX in range(colorDistance.shape[1]):
        for centers in range(noOfClusters):
            colorDistancecolumns.append(colorDistance[centers][imgX])

        minValIndexLoc = colorDistancecolumns.index(min(colorDistancecolumns))

        ptsClassified[minValIndexLoc][imgX]= 1
        colorDistancecolumns = []
    return ptsClassified

def reCalculateMean(ptsClassified, image,colorDistance, noOfClusters, clusterCenters,prevClusterCenters):
    clusterCenters = []
    newMean = []
    print(type(newMean))
    print(prevClusterCenters.shape)
    for cluster in range(noOfClusters):
        for imgX in range(colorDistance.shape[1]):
            if ptsClassified[cluster][imgX] == 1:
                # print(colorDistance[imgX])
                newMean.append(image[imgX])
        if not newMean:
            newMean.append(prevClusterCenters[cluster])
        mean = np.mean(newMean, axis= 0)
        clusterCenters.append(mean)
    return clusterCenters


