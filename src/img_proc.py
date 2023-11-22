import cv2
import numpy as np

class ImgProc:

    def convexHull(src_img_path, dest_img_path = 'img/ConvexHull.jpg'):
        # Load the image
        img = cv2.imread(src_img_path)
        # Convert it to greyscale
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey,(11,11),sigmaX=500,sigmaY=500)
        cv2.imwrite('img/blur.jpg',blur)
        # Threshold the image
        #ret, thresh = cv2.threshold(blur,190,255,cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
        thresh = (255-thresh)

        cv2.imwrite('img/thresh.jpg',thresh)
        # Find the contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img1 = img.copy()

        cv2.drawContours(img1, contours, -1, (0,255,0), 3)
        cv2.imwrite('img/contours.jpg',img1)
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
    
    def edge_lines(src_img_path,dest_img_path = 'img/lines.jpg'):
        img = cv2.imread(src_img_path)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey,(11,11),sigmaX=500,sigmaY=500)
        edges = cv2.Canny(blur, 50, 100)
        cv2.imwrite('img/edges.jpg',edges)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

        for line in lines:
            rho, theta = line[0]
            
            a = np.cos(theta)
            b = np.sin(theta)

            x0 = a * rho
            y0 = b * rho
            x1 = (int(x0 + 2000*(-b)))
            y1 = (int(y0 + 2000*(a)))
            x2 = (int(x0 - 2000*(-b)))
            y2 = (int(y0 - 2000*(a)))
            
            cv2.line(img,(x1,y1),(x2,y2), (0,0,255),2)

        cv2.imwrite('img/lines.jpg',img)

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