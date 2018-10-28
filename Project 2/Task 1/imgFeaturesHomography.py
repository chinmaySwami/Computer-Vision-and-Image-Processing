# ****************** CVIP Project 2 Task 1 ***********************************
# Title          :- Image Features and Homography
# Author         :- Chinmay Prakash Swami
# *****************************************************************************

import cv2
import  numpy as np

# Task1.1
image1 = cv2.imread("Images/mountain1.jpg")
image1G = cv2.imread("Images/mountain1.jpg",cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("Images/mountain2.jpg")
image2G = cv2.imread("Images/mountain2.jpg",cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()

# Detect and compute keypoints for image 1
keyPointImage1, descriptorsImage1 = sift.detectAndCompute(image1,None)
img1 =cv2.drawKeypoints(image1G,keyPointImage1,image1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("task1_sift1.jpg",img1)

# Detect and compute keypoints for image 2
keyPointImage2, descriptorsImage2 = sift.detectAndCompute(image2,None)
img2 =cv2.drawKeypoints(image2G,keyPointImage2,image2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("task1_sift2.jpg",img2)

# Task 1.2
bestMatcherObject = cv2.BFMatcher()
matchedKeypoints = bestMatcherObject.knnMatch(descriptorsImage1,descriptorsImage2,k=2)

# Filter good matches
goodMatches = []
for m,n in matchedKeypoints:
    if m.distance < 0.75*n.distance:
        goodMatches.append(m)

#Plot the matched keypoints that satisfied the threshold
img3 = cv2.drawMatches(image1G,keyPointImage1,image2G,keyPointImage2,goodMatches,None,flags=2)
cv2.imwrite("task1_matches_knn.jpg",img3)
