import cv2
import numpy as np
# Load the image
img1 = cv2.imread('pressedkeyboard.png')
# Convert it to greyscale
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# Threshold the image
ret, thresh = cv2.threshold(img,190,255,cv2.THRESH_BINARY)
cv2.imshow("TreshHull", thresh)
# Find the contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# For each contour, find the convex hull and draw it
# on the original image.
hull = []
for i in range(len(contours)):
    for j in range(len(contours[i])):
        #np.append(hull,contours[i][j])
        hull.append(contours[i][j])
        #print(contours[i][j])
hull = np.array(hull)
hull = cv2.convexHull(hull)
    
print(hull)
cv2.drawContours(img, hull, -1, (255, 0, 0), 2)
# Display the final convex hull image
cv2.imshow('ConvexHull', img)
cv2.waitKey(0)