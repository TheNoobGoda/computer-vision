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
        img2 = img.copy()
        x,y,_ =  img.shape
        img_center = (y/2,x/2)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #remove image boreders
        resize = img[50:len(img)-50]
        resize2 = []

        for i in range(resize.shape[0]):
            resize2.append([])
            for j in range(50,resize.shape[1]-50):
                resize2[i].append(resize[i][j])

        resize2 = np.array(resize2) 
        
        #detect white keys
        thresh2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
        cv2.imwrite('img/results/thresh.jpg',thresh2)
        dilation = cv2.morphologyEx(thresh2,cv2.MORPH_OPEN,kernel=np.ones((7, 7), np.uint8))
        cv2.imwrite('img/results/open.jpg',dilation)
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = 5000
        valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

        white_keys = []
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            white_keys.append((x,y,w,h)) 
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        

        #detect black keys
        _,thresh = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)
        opned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel= np.ones((13, 13), np.uint8))
        closed = cv2.morphologyEx(opned, cv2.MORPH_CLOSE, kernel= np.ones((7, 7), np.uint8))
        edges = cv2.Canny(closed,50,150)
        lines = cv2.HoughLines(edges,1,np.pi/180,75)
        cv2.imwrite('img/results/edges.jpg',edges)

        hor_lines = []
        if lines is not None:
            for i in range(0, len(lines)):
                theta = lines[i][0][1]
                if not (theta > 1 and theta < 2): continue
                hor_lines.append(lines[i])
        
        final_line = float('inf')
        for line in hor_lines:
            value = line[0][0]
            if value> img_center[1] and value < final_line: final_line = value
        
        
        points = []
        for i in range(len(edges[int(img_center[1])])):
            if edges[int(img_center[1])][i] == 255: points.append(i)


        left = points[0:int(len(points)/2)]
        right = points[int(len(points)/2):len(points)]
        left = left[len(left)-10:10]
        right = right[0:10]


        black_keys= []
        for i in range(len(left)//2):
            cv2.rectangle(img, (int(left[2*i]), 0), (int(left[2*i+1]), int(final_line)), (0, 0, 255), 2)
            black_keys.append((int(left[2*i]), 0,int(left[2*i+1])-int(left[2*i]),int(final_line)))
        
        for i in range(len(right)//2):
            cv2.rectangle(img, (int(right[2*i]), 0), (int(right[2*i+1]),int(final_line)), (0, 0, 255), 2)
            black_keys.append((int(right[2*i]), 0,int(right[2*i+1])-int(right[2*i]),int(final_line)))

        cv2.imwrite(dest_img_path,img)

        return(black_keys,white_keys)
        

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



        
       