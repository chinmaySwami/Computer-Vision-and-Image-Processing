# ****************** CVIP Project 1 Task 2 ***********************************
# Title          :- Cursor Detection
# Libraries Used :- OpenCV, np for array
# Author         :- Chinmay Prakash Swami
# *****************************************************************************

import cv2
import numpy as np
from matplotlib import pyplot as plt

images = ['neg_1.jpg','neg_2.jpg','neg_3.jpg','neg_4.jpg','neg_5.jpg','neg_6.jpg','neg_8.jpg',
          'neg_9.jpg','neg_10.jpg','pos_1.jpg','pos_2.jpg','pos_3.jpg','pos_4.jpg','pos_5.jpg','pos_6.jpg','pos_7.jpg',
          'pos_8.jpg','pos_9.jpg','pos_10.jpg','pos_11.jpg','pos_12.jpg','pos_13.jpg','pos_14.jpg','pos_15.jpg',]
for i in range(len(images)):
    img_name = "task3_images/"+ images[i]
    img = cv2.imread(img_name, 0)
    template = cv2.imread("task3_images/template_3.png", 0)

    # resized the template since it was quite big
    template = template[::4,::4]

    print(template.shape)
    width = template.shape[1]
    height = template.shape[0]
    print(width,height)

    # cv2.blur was used didnt work
    blurredImage = cv2.GaussianBlur(img, (3, 3), 2)   # didnt work
    # template = cv2.GaussianBlur(template, (1, 1), 0)  # didnt work

    imgEdges = cv2.Laplacian(blurredImage, cv2.CV_32F)
    templateEdges = cv2.Laplacian(template, cv2.CV_32F)

    result = cv2.matchTemplate(imgEdges, templateEdges, cv2.TM_CCORR_NORMED)
    threshold = 0.4
    templateLocation = []

    # Finding results that qualify the threshold
    for resrow in range(0,len(result)):
        for rescol in range(0,len(result[0])):
            if result[resrow][rescol] >= threshold:
                templateLocation.append([resrow, rescol])

    # Plotting the rectangles on the pointers detected
    for i in range(0,len(templateLocation)):
        x = templateLocation[i][0]
        y = templateLocation[i][1]
        cv2.rectangle(img, (y, x),(y+width,x+height), (0, 0, 0), 2)

    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ************************************* Bonus Part *********************************************
imagesBonusT1 = ['t1_1.jpg','t1_2.jpg','t1_3.jpg','t1_4.jpg','t1_5.jpg','t1_6.jpg']

for i in range(len(imagesBonusT1)):
    img_name = "task3_bonus/"+ imagesBonusT1[i]
    img = cv2.imread(img_name, 0)
    template = cv2.imread("task3_bonus/template_1.png", 0)

    # resized the template since it was quite big
    template = template[::4,::4]

    print(template.shape)
    width = template.shape[1]
    height = template.shape[0]
    print(width,height)

    # cv2.blur was used didnt work
    blurredImage = cv2.GaussianBlur(img, (3, 3), 2)   # didnt work
    # template = cv2.GaussianBlur(template, (1, 1), 0)  # didnt work

    imgEdges = cv2.Laplacian(blurredImage, cv2.CV_32F)
    templateEdges = cv2.Laplacian(template, cv2.CV_32F)

    result = cv2.matchTemplate(imgEdges, templateEdges, cv2.TM_CCORR_NORMED)
    threshold = 0.4
    templateLocation = []

    # Finding results that qualify the threshold
    for resrow in range(0,len(result)):
        for rescol in range(0,len(result[0])):
            if result[resrow][rescol] >= threshold:
                templateLocation.append([resrow, rescol])

    # Plotting the rectangles on the pointers detected
    for i in range(0,len(templateLocation)):
        x = templateLocation[i][0]
        y = templateLocation[i][1]
        cv2.rectangle(img, (y, x),(y+width,x+height), (0, 0, 0), 2)

    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imagesBonusT2 = ['t2_1.jpg','t2_2.jpg','t2_3.jpg','t2_4.jpg','t2_5.jpg','t2_6.jpg']

for i in range(len(imagesBonusT2)):
    img_name = "task3_bonus/"+ imagesBonusT2[i]
    img = cv2.imread(img_name, 0)
    template = cv2.imread("task3_bonus/template_2.png", 0)

    # resized the template since it was quite big
    template = template[::4, ::4]

    print(template.shape)
    width = template.shape[1]
    height = template.shape[0]
    print(width,height)

    # cv2.blur was used didnt work
    blurredImage = cv2.GaussianBlur(img, (3, 3), 1)   # didnt work
    # template = cv2.GaussianBlur(template, (1, 1), 0)  # didnt work

    imgEdges = cv2.Laplacian(blurredImage, cv2.CV_32F)
    templateEdges = cv2.Laplacian(template, cv2.CV_32F)

    result = cv2.matchTemplate(imgEdges, templateEdges, cv2.TM_CCORR_NORMED)
    threshold = 0.4
    templateLocation = []

    # Finding results that qualify the threshold
    for resrow in range(0,len(result)):
        for rescol in range(0,len(result[0])):
            if result[resrow][rescol] >= threshold:
                templateLocation.append([resrow, rescol])

    # Plotting the rectangles on the pointers detected
    for i in range(0,len(templateLocation)):
        x = templateLocation[i][0]
        y = templateLocation[i][1]
        cv2.rectangle(img, (y, x),(y+width,x+height), (0, 0, 0), 2)

    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imagesBonusT3 = ['t3_1.jpg','t3_2.jpg','t3_3.jpg','t3_4.jpg','t3_5.jpg','t3_6.jpg']

for i in range(len(imagesBonusT3)):
    img_name = "task3_bonus/"+ imagesBonusT3[i]
    img = cv2.imread(img_name, 0)
    template = cv2.imread("task3_bonus/template_3.png", 0)

    # resized the template since it was quite big
    template = template[::4, ::4]

    print(template.shape)
    width = template.shape[1]
    height = template.shape[0]
    print(width,height)

    # cv2.blur was used didnt work
    blurredImage = cv2.GaussianBlur(img, (3, 3), 1)   # didnt work
    # template = cv2.GaussianBlur(template, (1, 1), 0)  # didnt work

    imgEdges = cv2.Laplacian(blurredImage, cv2.CV_32F)
    templateEdges = cv2.Laplacian(template, cv2.CV_32F)

    result = cv2.matchTemplate(imgEdges, templateEdges, cv2.TM_CCORR_NORMED)
    threshold = 0.4
    templateLocation = []

    # Finding results that qualify the threshold
    for resrow in range(0,len(result)):
        for rescol in range(0,len(result[0])):
            if result[resrow][rescol] >= threshold:
                templateLocation.append([resrow, rescol])

    # Plotting the rectangles on the pointers detected
    for i in range(0,len(templateLocation)):
        x = templateLocation[i][0]
        y = templateLocation[i][1]
        cv2.rectangle(img, (y, x),(y+width,x+height), (0, 0, 0), 2)

    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()