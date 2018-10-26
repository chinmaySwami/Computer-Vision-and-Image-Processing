# ****************** CVIP Project 2 Task 1 ***********************************
# Title          :- Image Features and Homography
# Author         :- Chinmay Prakash Swami
# *****************************************************************************

import cv2
import  numpy as np

image1 = cv2.imread("Images/mountain1.jpg")
image1G = cv2.imread("Images/mountain1.jpg",cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("Images/mountain2.jpg",cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(image1,None)
img=cv2.drawKeypoints(image1G,kp,image1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("test",img)
cv2.waitKey(0)            # waits for user to press any key
cv2.destroyAllWindows()   # closes all output windows when a key is pressed