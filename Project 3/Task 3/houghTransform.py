import cv2
import numpy as np
from functions import padd, detectEdges, hough_line, hough_lines_draw,detect_lines
import math

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

# **********************START:Thresholding the edge detected image *******************
img_opyNB = np.zeros_like(img_opyN)
img_opxNB = np.zeros_like(img_opxN)

for imrow in range(0, len(img_opyN)):
    for imcol in range(0, len(img_opyN[0])):
        if img_opyN[imrow][imcol] > 30:
            img_opyNB[imrow][imcol] = 255

for imrow in range(0, len(img_opxN)):
    for imcol in range(0, len(img_opxN[0])):
        if img_opxN[imrow][imcol] > 88:
            img_opxNB[imrow][imcol] = 255

cv2.imwrite("ThresholdedY.jpg",img_opyNB)
cv2.imwrite("ThresholdedX.jpg",img_opxNB)
# **********************END:Thresholding the edge detected image *******************

print("Calculating for Diagonal Lines")
accumulator, thetas, rhos = hough_line(img_opyNB)
cv2.imwrite("accumulator.jpg", accumulator)
detect_lines(img, 9, accumulator, rhos, thetas)

print("Calculating for Horizontal Lines")
accumulator, thetas, rhos = hough_line(img_opxNB)
cv2.imwrite("accumulator.jpg", accumulator)
detect_lines(img, 8, accumulator, rhos, thetas)
