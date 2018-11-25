# ****************** CVIP Project 2 Task 2 ***********************************
# Title          :- Image Features and Homography
# Author         :- Chinmay Prakash Swami
# Refrences      :- Open CV Tutorials
# *****************************************************************************
import numpy as np;
import cv2
from matplotlib import pyplot as plt
from functions import generateMatches, generateKeypoints, drawLines
import random

# Task1.1
image1 = cv2.imread("Images/tsucuba_left.png")
image1G = cv2.imread("Images/tsucuba_left.png", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("Images/tsucuba_right.png")
image2G = cv2.imread("Images/tsucuba_right.png", cv2.IMREAD_GRAYSCALE)

# Detect and compute keypoints for image 1 and image 2
keyPointImage1, descriptorsImage1, keyPointImage2, descriptorsImage2 = generateKeypoints(image1G, image2G)

img1 =cv2.drawKeypoints(image1G, keyPointImage1, image1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("task2_sift1.jpg", img1)

img2 =cv2.drawKeypoints(image2G, keyPointImage2, image2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("task2_sift2.jpg", img2)

goodMatches = generateMatches(descriptorsImage1, descriptorsImage2)

# Plot the matched keypoints that satisfied the threshold
img3 = cv2.drawMatches(image1G, keyPointImage1, image2G, keyPointImage2, goodMatches, None, flags=2)
cv2.imwrite("task2_matches_knn.jpg", img3)

sourcePoints = np.float32([keyPointImage1[m.queryIdx].pt for m in goodMatches])
destinationPoints = np.float32([keyPointImage2[m.trainIdx].pt for m in goodMatches])

# Generate Fundamental matrix
fundamentalMatrix, inOut = cv2.findFundamentalMat(sourcePoints,destinationPoints,cv2.RANSAC)

print("Fundamental matrix: \n",fundamentalMatrix)
# Selecting only inliers
sourcePoints = sourcePoints[inOut.ravel()==1]
destinationPoints = destinationPoints[inOut.ravel()==1]

# Selecting 10 random points
sourcePointsSmall = []
destinationPointsSmall = []
for i in range(10):
    index = random.randint(0,min(len(sourcePoints)-1,len(destinationPoints)-1))
    sourcePointsSmall.append(sourcePoints[index])
    destinationPointsSmall.append(destinationPoints[index])

sourcePointsSmall = np.asarray(sourcePointsSmall)
destinationPointsSmall = np.asarray(destinationPointsSmall)

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(destinationPointsSmall.reshape(-1,1,2), 2,fundamentalMatrix)
lines1 = lines1.reshape(-1,3)
epiRight,img6 = drawLines(image1G,image2G,lines1,sourcePointsSmall,destinationPointsSmall)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(sourcePointsSmall.reshape(-1,1,2), 1,fundamentalMatrix)
lines2 = lines2.reshape(-1,3)
epiLeft,img4 = drawLines(image2G,image1G,lines2,destinationPointsSmall,sourcePointsSmall)

cv2.imwrite("task2_epi_right.jpg",epiRight)
cv2.imwrite("task2_epi_left.jpg",epiLeft)

#  Task 1.5
stereo = cv2.StereoSGBM_create(minDisparity=-20, numDisparities = 80,blockSize=17)
disparities = stereo.compute(image1G,image2G)

plt.imshow(disparities,'gray')
plt.imsave("task2_disparity.png",disparities,cmap ='gray')
plt.show()