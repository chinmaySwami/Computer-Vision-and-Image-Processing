import cv2
import numpy as np

def padd(img):

    imgL = [[0 for imrow in range(len(img[0])+2)]for imcol in range(len(img)+2)]
    print(len(img), len(img[0]), len(imgL), len(imgL[0]))
    for imrow in range(1, len(imgL)-1):
        for imcol in range(1, len(imgL[0])-1):
            imgL[imrow][imcol] = img[imrow-1][imcol-1]
    imgL = np.asarray(imgL)
    print(type(imgL))
    return imgL

def findMax(img):
    maxVal = img[0][0]
    for imrow in range(len(img)):
        for imcol in range(len(img[0])):
                if maxVal <= img[imrow][imcol]:
                    maxVal = img[imrow][imcol]
    return maxVal

def findAbs(img):
    for imrow in range(len(img)):
        for imcol in range(len(img[0])):
                if img[imrow][imcol] < 0:
                    img[imrow][imcol] *= -1
    return img

def detectEdges(img, sobelx, sobely, img_opx, img_opy, img_combined):
    print("Performing Convolution")
    # Perform convolutions
    for imrow in range(0, len(img)):
        for imcol in range(0, len(img[0])):
            pixvalx = 0
            pixvaly = 0
            if imrow > 0 and imcol > 0 and imrow < len(img)-1 and imcol < len(img[0])-1:
                pixvalx = ((sobelx[0][0]*img[imrow-1][imcol-1]) + (sobelx[0][1]*img[imrow-1][imcol]) +
                           (sobelx[0][2]*img[imrow-1][imcol+1]) +
                           (sobelx[1][0]*img[imrow][imcol-1]) + (sobelx[1][1]*img[imrow][imcol]) +
                           (sobelx[1][2]*img[imrow][imcol+1]) +
                           (sobelx[2][0]*img[imrow+1][imcol-1]) + (sobelx[2][1]*img[imrow+1][imcol]) +
                           (sobelx[2][2]*img[imrow+1][imcol+1]))

                pixvaly = ((sobely[0][0]*img[imrow-1][imcol-1]) + (sobely[0][1]*img[imrow-1][imcol]) +
                           (sobely[0][2]*img[imrow-1][imcol+1]) +
                           (sobely[1][0]*img[imrow][imcol-1]) + (sobely[1][1]*img[imrow][imcol]) +
                           (sobely[1][2]*img[imrow][imcol+1]) +
                           (sobely[2][0]*img[imrow+1][imcol-1]) + (sobely[2][1]*img[imrow+1][imcol]) +
                           (sobely[2][2]*img[imrow+1][imcol+1]))

            img_opx[imrow][imcol] = pixvalx  # calculating for horizontal edges
            img_opy[imrow][imcol] = pixvaly # calculating for vertical edges
            # combining horizontal and vertical image referred wiki page of sobel for the below formula
            img_combined[imrow][imcol] = (pixvalx**2 + pixvaly**2)**(1/2)

    img_opxN = findAbs(img_opx) / findMax(abs(img_opx))
    img_opxN = img_opxN * 255
    img_opxN = np.asarray(img_opxN,dtype=np.uint8)
    # cv2.imwrite('X_direction.jpg', img_opxN)

    img_opyN = findAbs(img_opy) / findMax(abs(img_opy))
    img_opyN = img_opyN * 255
    img_opyN = np.asarray(img_opyN, dtype=np.uint8)
    # cv2.imwrite('Y_direction.jpg', img_opyN)

    img_combinedN = findAbs(img_combined) / findMax(abs(img_combined))
    # img_combinedN = img_combinedN * 255
    # img_combinedN = np.asarray(img_combinedN, dtype=np.uint8)
    # cv2.imwrite('Combined.jpg', img_combinedN)

    return img_opxN, img_opyN