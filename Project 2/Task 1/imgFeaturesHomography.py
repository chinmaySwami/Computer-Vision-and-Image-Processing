# ****************** CVIP Project 2 Task 1 ***********************************
# Title          :- Image Features and Homography
# Author         :- Chinmay Prakash Swami
# Refrences      :- Open CV Tutorials
#                   http://answers.opencv.org/question/144252/perspective-transform-without-crop/
#                   To solve pixel loss issue
# *****************************************************************************
import numpy as np;
import cv2

from functions import generateKeypoints,generateMatches
import random

# Task1.1
image1 = cv2.imread("Images/mountain1.jpg")
image1G = cv2.imread("Images/mountain1.jpg",cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("Images/mountain2.jpg")
image2G = cv2.imread("Images/mountain2.jpg",cv2.IMREAD_GRAYSCALE)

height1,width1 = image1G.shape
height2,width2 = image2G.shape

# Detect and compute keypoints for image 1 and image 2
keyPointImage1, descriptorsImage1, keyPointImage2, descriptorsImage2 = generateKeypoints(image1G, image2G)

img1 =cv2.drawKeypoints(image1G,keyPointImage1,image1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("task1_sift1.jpg",img1)

img2 =cv2.drawKeypoints(image2G,keyPointImage2,image2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("task1_sift2.jpg",img2)

# Task 1.2
goodMatches = generateMatches(descriptorsImage1, descriptorsImage2)

# Plot the matched keypoints that satisfied the threshold
img3 = cv2.drawMatches(image1G, keyPointImage1, image2G, keyPointImage2, goodMatches, None, flags=2)
cv2.imwrite("task1_matches_knn.jpg", img3)

# Task 1.3 & 1.4
sourcePoints = np.float32([keyPointImage1[m.queryIdx].pt for m in goodMatches])
destinationPoints = np.float32([keyPointImage2[m.trainIdx].pt for m in goodMatches])

homographyMatrix, inOut = cv2.findHomography(sourcePoints, destinationPoints, cv2.RANSAC, 5.0)
print("Homography matrix H is: \n",homographyMatrix)
matchedMask = inOut.ravel().tolist() # ravel is used to flatten the inOut array

#  Calculate 10 random inliers
matchedMask2 = np.zeros(len(matchedMask))
matchedMask2 = matchedMask2.tolist()
count = 0
while count <= 10:
    index = random.randint(0, len(matchedMask)-1)
    if matchedMask[index] == 1:
        matchedMask2[index] = 1
        count += 1
# unique, counts = np.unique(matchedMask, return_counts=True)
# print("\n Inliers and Outliers Stats: ", dict(zip(unique, counts)))

parameters = dict(matchColor = (0, 200, 0), # draw matches in black color
                singlePointColor = None,
                matchesMask = matchedMask2, # draw only inliers
                flags = 2)

img5 = cv2.drawMatches(image1G, keyPointImage1, image2G, keyPointImage2, goodMatches, None, **parameters)
cv2.imwrite("task1_matches.jpg",img5)

#  Code to find corners after transformation is applied
#  Calculating the end points
endPoints = np.array([
    [0,0],
    [0,height1],
    [width1, height1],
    [width1,0]
])

corners = cv2.perspectiveTransform(np.float32([endPoints]), homographyMatrix)
# Find the bounding rectangle
x, y, boundedWidth, boundedHeight = cv2.boundingRect(corners)
# Translating -ve points which are causing loss of pixels into +Ve points
intermediateTransformationMatrix = np.array([
  [ 1, 0, -x ],
  [ 0, 1, -y ],
  [ 0, 0,  1 ]
])

homographyMatrix = intermediateTransformationMatrix.dot(homographyMatrix)
print("\n New Homography matrix H is: \n",homographyMatrix)

result1 = cv2.warpPerspective(image1G, homographyMatrix, (boundedWidth, boundedHeight), flags=cv2.INTER_CUBIC)
result1 = cv2.copyMakeBorder(result1,0,15,10,0,cv2.BORDER_CONSTANT)
cv2.imwrite("image1wrap.jpg",result1)
boundedHeight, boundedWidth = result1.shape

#  Creating a blank image
result = np.zeros((max(boundedHeight, height2), width2+width2), dtype=np.uint8)

result[:boundedHeight, :boundedWidth] = result1
result[boundedHeight-height2:, width1:] = image2G
cv2.imwrite("task1_pano.jpg",result)
