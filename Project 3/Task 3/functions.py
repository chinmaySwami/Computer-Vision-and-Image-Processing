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

def hough_line(img):
  # Rho and Theta ranges
  #  np.cos doesnt work with degrees hence it was changed to radians
  thetas = np.deg2rad(np.arange(-90.0, 90.0))
  imageWidth, imageHeight = img.shape
  # find max distance to avoid getting array out of bounds error
  diag_len = int(np.sqrt(imageWidth * imageWidth + imageHeight * imageHeight))
  rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)
  # Cache some resuable values
  cos_t = np.cos(thetas)
  sin_t = np.sin(thetas)
  num_thetas = len(thetas)
  # Hough accumulator array of theta vs rho
  accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
  # getting the X TY co-ordinates of the edges.
  y_idxs, x_idxs = np.nonzero(img)
  # Vote in the hough accumulator
  for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]
    for t_idx in range(num_thetas):
      # Calculate rho. diag_len is added for a positive index
      rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
      accumulator[rho, t_idx] += 1
  return accumulator, thetas, rhos

def hough_lines_draw(img, indicies, rhos, thetas):
    ''' A function that takes indicies a rhos table and thetas table and draws
        lines on the input images that correspond to these values. '''
    for i in range(len(indicies)):
        # reverse engineer lines from rhos and thetas
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite('output.jpg', img)

def detect_lines(image, noOfPeaks, acc, rhos, thetas):
    distHistorical = 0
    for i in range(noOfPeaks):
        arr = np.unravel_index(acc.argmax(), acc.shape)
        acc[arr[0]][arr[1]] = 0
        acc[arr[0]-20:arr[0]+20, arr[1]-20:arr[0]+20] = 0

        #acc[(i-h):(i-h)+5, (j-w):(j-w)+5] = 0
        rho = rhos[arr[0]]
        theta = thetas[arr[1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite('output.jpg', image)