import cv2
import numpy as np

def generateKeypoints(image1,image2):
    sift = cv2.xfeatures2d.SIFT_create()
    keyPoint1, descriptors1 = sift.detectAndCompute(image1, None)
    keyPoint2, descriptors2 = sift.detectAndCompute(image2, None)
    return keyPoint1,descriptors1,keyPoint2,descriptors2

def generateMatches(descriptor1,descriptor2):
    bestMatcherObject = cv2.BFMatcher()
    matchedKeypoints = bestMatcherObject.knnMatch(descriptor1, descriptor2, k=2)

    # Filter good matches: Lowe's Ratio test to determine high quality features
    goodMatches = []
    for m, n in matchedKeypoints:
        if m.distance < 0.75 * n.distance:
            goodMatches.append(m)

    return goodMatches

def drawLines(img1, img2, lines, sourcePoints, destinationPoints):
    r, c = img1.shape
# This is done so that we can show color lines and points on the image
# creating list of colors
    colors = ((109, 193, 120),
             (149, 50, 148),
             (196, 239, 3),
             (92, 195, 53),
             (28, 64, 91),
             (127, 92, 87),
             (90, 67, 49),
             (81, 74, 60),
             (128, 26, 209),
             (147, 36, 145))
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    colorIndex = 0
    for r, sourcePoints, destinationPoints in zip(lines, sourcePoints, destinationPoints):
# to draw line we need two points hence we calculate X0,Y0 and x1,y1
        color = colors[colorIndex]
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
# Draw the line with the above points
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(sourcePoints), 5, color, -1)
        img2 = cv2.circle(img2, tuple(destinationPoints), 5, color, -1)
        colorIndex += 1
    return img1, img2

