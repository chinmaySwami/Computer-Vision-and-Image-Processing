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
    for i,(m, n) in enumerate(matchedKeypoints):
        if m.distance < 0.75 * n.distance:
            goodMatches.append(m)
    return goodMatches