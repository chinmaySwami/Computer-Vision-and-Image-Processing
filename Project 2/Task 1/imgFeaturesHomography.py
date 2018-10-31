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

print(goodMatches)
# Plot the matched keypoints that satisfied the threshold
img3 = cv2.drawMatches(image1G,keyPointImage1,image2G,keyPointImage2,goodMatches,None,flags=2)
cv2.imwrite("task1_matches_knn.jpg",img3)

# Task 1.3 & 1.4

sourcePoints = np.float32([keyPointImage1[m.queryIdx].pt for m in goodMatches])
destinationPoints = np.float32([keyPointImage2[m.trainIdx].pt for m in goodMatches])

# for m in goodMatches:
#     sourcePoints = np.float32(keyPointImage1[m.queryIndex].pt)
# sourcePoints.reshape(-1,1,2)
#
# for m in goodMatches:
#     destinationPoints = np.float32(keyPointImage2[m.queryIndex].pt)
# destinationPoints.reshape(-1,1,2)

homographyMatrix, inOut = cv2.findHomography(sourcePoints,destinationPoints,cv2.RANSAC)
print("homography matrix H is: \n",homographyMatrix)
print(type(inOut),inOut.shape)
matchedMask = inOut.ravel().tolist() # ravel is used to flatten the inOut array

unique, counts = np.unique(matchedMask, return_counts=True)
print(dict(zip(unique, counts)))

height1,width1 = image1G.shape
height2,width2 = image2G.shape

print(height1,width1)
print(height2,width2)

# points = np.float32([ [0,0],[0,height-1],[width-1,height-1],[width-1,0] ])
# distance = cv2.perspectiveTransform(points,homographyMatrix)

parameters = dict(matchColor = (0,200,0), # draw matches in black color
                   singlePointColor = None,
                   matchesMask = matchedMask, # draw only inliers
                   flags = 2)

img5 = cv2.drawMatches(image1G,keyPointImage1,image2G,keyPointImage2,goodMatches,None,**parameters)
cv2.imwrite("task1_matches.jpg",img5)

result = cv2.warpPerspective(image2G, homographyMatrix,
			(image1G.shape[1] + image2G.shape[1], image1G.shape[0]))

cv2.imwrite("Beforeedit.jpg",result)
result[0:image2G.shape[0], 0:image2G.shape[1]] = image1G
# result12 = cv2.warpPerspective(image1G, homographyMatrix,(image2G.shape[1], image2G.shape[0]))
# cv2.imwrite("task1_pano12.jpg",result12)

# result21 = cv2.warpPerspective(image2G, homographyMatrix,(image1G.shape[1], image1G.shape[0]))
cv2.imwrite("afteredit.jpg",result)
