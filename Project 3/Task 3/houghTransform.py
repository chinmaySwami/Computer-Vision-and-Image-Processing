import cv2
import numpy as np
from functions import padd,detectEdges

# **********************Start:Performing Edge detection********************************
img = cv2.imread('Images/hough.jpg', cv2.IMREAD_GRAYSCALE)
img = padd(img)
img_opx = img.copy()
img_opy = img.copy()
img_combined = img.copy()

# Sobely is sobel kernel for Y axis which has been flipped left->right and top->bottom
sobely = [[-1, -2, -1],
          [0, 0, 0],
          [1, 2, 1]]

# Sobelx is sobel kernel for X axis which has been flipped left->right and top->bottom
sobelx = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]

img_opxN, img_opyN = detectEdges(img, sobelx, sobely, img_opx, img_opy, img_combined)
# **********************END:Performing Edge detection********************************

img_opyNB = np.zeros_like(img_opyN)
for imrow in range(0, len(img_opyN)):
    for imcol in range(0, len(img_opyN[0])):
        if img_opyN[imrow][imcol] > 32:
            img_opyNB[imrow][imcol] = 255

cv2.imwrite("Thresholded.jpg",img_opyNB)