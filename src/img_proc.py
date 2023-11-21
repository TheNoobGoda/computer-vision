import cv2
import numpy as np

class ImgProc:

    def convexHull(src_img_path, dest_img_path = 'img/ConvexHull.jpg'):
        # Load the image
        img1 = cv2.imread(src_img_path)
        # Convert it to greyscale
        img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # Threshold the image
        ret, thresh = cv2.threshold(img,190,255,cv2.THRESH_BINARY)
        # Find the contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # For each contour, find the convex hull and draw it
        # on the original image.
        hull = []
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                hull.append(contours[i][j])
        hull = np.array(hull)
        hull = cv2.convexHull(hull)
    
        x,y,w,h = cv2.boundingRect(hull)
        cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
        # Display the final convex hull image
        cv2.imwrite(dest_img_path, img1)
        return hull

    def get_keyboard(src_img_path,hull, dest_img_path = 'img/cropped_keyboard.jpg'):
        img = cv2.imread(src_img_path)
        x,y,w,h = cv2.boundingRect(hull)
        crop_img = img[y:y+h, x:x+w]
        cv2.imwrite(dest_img_path, crop_img)
        return crop_img