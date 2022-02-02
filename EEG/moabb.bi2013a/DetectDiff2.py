# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 17:10:49 2022

@author: antona
"""

from skimage.metrics import structural_similarity
import argparse
import imutils
import cv2
 
 
# load the two input images
imageA = cv2.imread('c:\\temp\\img1AverageRotated.png')
imageB = cv2.imread('c:\\temp\\img2AverageRotated.png')
 
# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = structural_similarity(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))
# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,
 cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
 cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)
# loop over the contours
i=1
for c in cnts:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    print(i)
    (x, y, w, h) = cv2.boundingRect(c)
    print(x,y,w,h)
    
    #1 2 3 5 6 7
    if i == 1 or i == 2 or i == 3 or i == 5 or i == 6 or i == 7:
       
        #print(x,y,w,h)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    
    i = i + 1
    
# show the output images
cv2.imwrite("c:\\temp\\diff2\\Original.png", imageA)
cv2.imwrite("c:\\temp\\diff2\\Modified.png", imageB)
cv2.imwrite("c:\\temp\\diff2\\Diff2.png", diff)
cv2.imwrite("c:\\temp\\diff2\\Thresh.png", thresh)
print("Done")