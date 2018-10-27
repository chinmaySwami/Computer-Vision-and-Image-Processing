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
kp1, des1 = sift.detectAndCompute(image1,None)
img1 =cv2.drawKeypoints(image1G,kp1,image1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("Task1-1output1.jpg",img1)

kp2, des2 = sift.detectAndCompute(image2,None)
img2 =cv2.drawKeypoints(image2G,kp2,image2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("Task1-1output2.jpg",img2)

# Task 1.2
