import cv2
import numpy as np

class ImgProc:

    def convexHull(src_img_path, dest_img_path = 'img/results/ConvexHull.jpg'):
        # Load the image
        img = cv2.imread(src_img_path)
        # Convert it to greyscale
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey,(7,7),sigmaX=500,sigmaY=500)
        # Threshold the image
        ret, thresh = cv2.threshold(blur,190,255,cv2.THRESH_BINARY)
        cv2.imwrite('img/results/thresh.jpg',thresh)
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

    def crop_img(src_img_path,hull, dest_img_path = 'img/results/cropped_keyboard.jpg'):
        img = cv2.imread(src_img_path)
        x,y,w,h = cv2.boundingRect(hull)
        crop_img = img[y:y+h, x:x+w]
        cv2.imwrite(dest_img_path, crop_img)
        return crop_img
    
    def find_keys(src_img_path, dest_img_path = 'img/results/keys.jpg'):
        img = cv2.imread(src_img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resize = img[50:len(img)-50]
        resize2 = []

        for i in range(resize.shape[0]):
            resize2.append([])
            for j in range(50,resize.shape[1]-50):
                resize2[i].append(resize[i][j])

        resize2 = np.array(resize2) 

        #cv2.imwrite('img/results/resize.jpg',resize2)

        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        _,thresh = cv2.threshold(resize2,127,255,cv2.THRESH_BINARY)
        #cv2.imwrite('img/results/thresh.jpg',thresh)

        kernel = np.ones((13, 13), np.uint8)
        opned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        #cv2.imwrite('img/results/open.jpg',opned)

        kernel = np.ones((7, 7), np.uint8)
        closed = cv2.morphologyEx(opned, cv2.MORPH_CLOSE, kernel)
        #cv2.imwrite('img/results/close.jpg',closed)

        edges = cv2.Canny(closed,50,150)
        #cv2.imwrite('img/results/canny.jpg',edges)

        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get the bounding box of the key
            x, y, w, h = cv2.boundingRect(approx)
            x +=50
            y +=50
            
            # Draw the bounding box on the original image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        thresh2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
        cv2.imwrite('img/results/thresh.jpg',thresh2)
        kernel = np.ones((7, 7), np.uint8)
        #dilation = cv2.erode(thresh2,kernel,iterations = 3)
        dilation = cv2.morphologyEx(thresh2,cv2.MORPH_OPEN,kernel)
        cv2.imwrite('img/results/open.jpg',dilation)
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = 5000  # adjust as needed
        valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        cv2.imwrite(dest_img_path,img)
        

    def find_piano(image_path,dest_img_path = 'img/results/cropped_keyboard.jpg'):
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



        
       