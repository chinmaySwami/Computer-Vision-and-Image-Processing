import cv2
import numpy as np
import math

def padd(img):

    imgL = [[0 for imrow in range(len(img[0])+2)]for imcol in range(len(img)+2)]
    print(len(img), len(img[0]), len(imgL), len(imgL[0]))
    for imrow in range(1, len(imgL)-1):
        for imcol in range(1, len(imgL[0])-1):
            imgL[imrow][imcol] = img[imrow-1][imcol-1]
    imgL = np.asarray(imgL)
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

    return img_opxN, img_opyN
def carryoutVoting(image,cosValues, sinValues, numberOfThetaValues,diagonalLength):
    # Generating accumulator array
    accumulator = np.zeros((2 * diagonalLength, numberOfThetaValues), dtype=np.uint8)
    # getting the X & Y co-ordinates of the edges. which are edges hence nonZero function is used
    yIndexes, xIndexes = np.nonzero(image)
    # Writing the code for the voting functionality
    for i in range(len(xIndexes)):
        # getting x and y values 1 at a time
        x = xIndexes[i]
        y = yIndexes[i]
        for index in range(numberOfThetaValues):
            # Calculate rho. diag_len is added for a positive index
            rho = int(round(x * cosValues[index] + y * sinValues[index]) + diagonalLength)
            accumulator[rho, index] += 1
    return accumulator


# Referred to https://alyssaq.github.io/2014/understanding-hough-transform/ for getting an idea of what
# needs to be done
def hough_line(img):
  # Rho and Theta ranges np.cos doesnt work with degrees hence it was changed to radians
  thetas = np.deg2rad(np.arange(-90.0, 90.0))
  imageWidth, imageHeight = img.shape
  # find max distance to avoid getting array out of bounds error
  diagonalLength = int(np.sqrt(imageWidth * imageWidth + imageHeight * imageHeight))
  rhos = np.linspace(-diagonalLength, diagonalLength, (diagonalLength * 2.0))
  # Generating cos and sin values which will we will be using in voting code.
  cosValues = np.cos(thetas)
  sinValues = np.sin(thetas)
  numberOfThetaValues = len(thetas)
  accumulator = carryoutVoting(img, cosValues, sinValues, numberOfThetaValues, diagonalLength)
  return accumulator, thetas, rhos

def drawLinesOnImage(image, noOfPeaks, acc, rhos, thetas, isItRed):
    # Here I have implemented a mask which clears the surrounding 10 pixels of the max point detected
    # by setting the co-ordinate values to 0. This is done to avoid plotting multiple lines for the same line.

    distHistorical = 0
    for i in range(noOfPeaks):
        #  Finding the Max value
        arr = np.unravel_index(acc.argmax(), acc.shape)
        acc[arr[0]][arr[1]] = 0
        # Making surrounding pixel value of the max pixel as 0
        acc[arr[0]-20:arr[0]+20, arr[1]-20:arr[0]+20] = 0
        rho = rhos[arr[0]]
        theta = thetas[arr[1]]
        cosTheta = np.cos(theta)
        sinTheta = np.sin(theta)
        x0 = cosTheta*rho
        y0 = sinTheta*rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 850*(-sinTheta))
        y1 = int(y0 + 850*(cosTheta))
        x2 = int(x0 - 850*(-sinTheta))
        y2 = int(y0 - 850*(cosTheta))

        if isItRed:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 10), 2)
            cv2.imwrite('red_line.jpg', image)
        else:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 10), 2)
            cv2.imwrite('blue_lines.jpg', image)