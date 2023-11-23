import cv2
import numpy as np

class ImgProc:

    def convexHull(src_img_path, dest_img_path = 'img/ConvexHull.jpg'):
        # Load the image
        img = cv2.imread(src_img_path)
        # Convert it to greyscale
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey,(7,7),sigmaX=500,sigmaY=500)
        # Threshold the image
        ret, thresh = cv2.threshold(blur,190,255,cv2.THRESH_BINARY)
        cv2.imwrite('img/thresh.jpg',thresh)
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
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # Display the final convex hull image
        cv2.imwrite(dest_img_path, img)
        return hull

    def crop_img(src_img_path,hull, dest_img_path = 'img/cropped_keyboard.jpg'):
        img = cv2.imread(src_img_path)
        x,y,w,h = cv2.boundingRect(hull)
        crop_img = img[y:y+h, x:x+w]
        cv2.imwrite(dest_img_path, crop_img)
        return crop_img
    
    def img_thresh(src_img_path,threshold = 127, dest_img_path = 'img/img_thresh.jpg'):
        img = cv2.imread(src_img_path)
        ret, thresh = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
        thresh[np.all(thresh == (255,255,255),axis=-1)] = (255,0,0)
        thresh[np.all(thresh == (0,0,0),axis=-1)] = (0,255,0)
        cv2.imwrite(dest_img_path,thresh)

    def find_piano(image_path,dest_img_path = 'img/cropped_keyboard.jpg'):
        # Read the input image
        original_image = cv2.imread(image_path)
        blur = cv2.GaussianBlur(original_image,(15,15),50)

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Define a mask for the color range of white keys
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

        piano_image = cv2.bitwise_and(original_image, original_image, mask=mask_white)
        cv2.imshow('Piano Image', piano_image)

        # Find contours in the white mask
        contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the bounding box of the combined contours
        x, y, w, h = cv2.boundingRect(np.vstack(contours))

        # Crop the original image to the bounding box
        cropped_image = original_image[y:y+h, x:x+w]

        cv2.imwrite(dest_img_path, cropped_image)

        return cropped_image