import cv2
import numpy as np

class ImgProc:
    
    def find_keys(image,save_imgs):
        img2 = image.copy()
        x,y,_ =  image.shape
        img_center = ((y/2)-10,x/2)
        
        #detect white keys

        #find contours in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
        dilation = cv2.morphologyEx(thresh2,cv2.MORPH_OPEN,kernel=np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #filter contours based on area
        min_contour_area = 2000
        valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

        #transform contours in coordinates
        white_keys = []
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if y != 0:
                h += y
                y = 0
            white_keys.append((x,y,w,h)) 
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        

        #detect black keys

        #find lines in image
        blur = cv2.GaussianBlur(img2,[7,7],1)
        _,thresh = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
        opned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel= np.ones((13, 13), np.uint8))
        closed = cv2.morphologyEx(opned, cv2.MORPH_CLOSE, kernel= np.ones((7, 7), np.uint8))
        edges = cv2.Canny(closed,50,150)
        lines = cv2.HoughLines(edges,1,np.pi/180,75)
        
        #filter lines to only horizontal
        hor_lines = []
        if lines is not None:
            for i in range(0, len(lines)):
                theta = lines[i][0][1]
                if not (theta > 1 and theta < 2): continue
                hor_lines.append(lines[i])
        
        #find the line closeste to the center of the image
        final_line = float('inf')
        for line in hor_lines:
            value = line[0][0]
            if value> img_center[1] and value < final_line: final_line = value
        
        if hor_lines == []:
            final_line = x
        
        #detect points in the edges of the image along the horizontal line found
        points = []
        for i in range(len(edges[int(img_center[1])])):
            if edges[int(img_center[1])][i] == 255: points.append(i)
       
        #filter out points that are too close
        point = 0
        for i in points:
            if i < point+5: 
                points.remove(i)
                continue
            point = i

        #divide points into left of the center and right of the center
        left = []
        right = []

        for i in  points:
            if i < img_center[0]: left.append(i)
            else: right.append(i)

        #filter the points found to only ten in each side of the image
        left = left[len(left)-10:10]
        right = right[0:10]


        #3xtract rectangles around detected black keys based on the points found
        black_keys= []
        for i in range(len(left)//2):
            cv2.rectangle(image, (int(left[2*i]), 0), (int(left[2*i+1]), int(final_line)), (0, 0, 255), 2)
            black_keys.append((int(left[2*i]), 0,int(left[2*i+1])-int(left[2*i]),int(final_line)))
        
        for i in range(len(right)//2):
            cv2.rectangle(image, (int(right[2*i]), 0), (int(right[2*i+1]),int(final_line)), (0, 0, 255), 2)
            black_keys.append((int(right[2*i]), 0,int(right[2*i+1])-int(right[2*i]),int(final_line)))

        #save intermediate images for debugging
        if save_imgs:
            cv2.imwrite('img/results/keys.jpg',image)
            cv2.imwrite('img/results/find_keys_thresh2.jpg',thresh2)
            cv2.imwrite('img/results/find_keys_dilation.jpg',dilation)
            cv2.imwrite('img/results/find_keys_edges.jpg',edges)
            cv2.imwrite('img/results/find_keys_blur.jpg',blur)
            cv2.imwrite('img/results/find_keys_thresh.jpg',thresh)
            cv2.imwrite('img/results/find_keys_opned.jpg',opned)
            cv2.imwrite('img/results/find_keys_closed.jpg',closed)

        return(black_keys,white_keys)
        

    def find_piano(image,save_imgs):
        blur = cv2.GaussianBlur(image,(15,15),50)
        hsv_image = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        #define a mask for the color range of white keys
        lower_white = np.array([0, 0, 220])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

        #find contours in the white mask
        contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #find the bounding box of the combined contours
        x, y, w, h = cv2.boundingRect(np.vstack(contours))

        #crop the original image to the bounding box
        cropped_image = image[y:y+h, x:x+w]

        #save intermediate images for debugging
        if save_imgs:
            cv2.imwrite('img/results/cropped_keyboard.jpg', cropped_image)
            cv2.imwrite('img/results/find_piano_blur.jpg', blur)
        
        return cropped_image, (x, y)
    
    def get_first_frame(src_vid_path,save_imgs):
        cap = cv2.VideoCapture(src_vid_path)

        #read the first frame
        _, img = cap.read()

        #remove this code after
        img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        #up to here

        #save image for debugging
        if save_imgs:
            cv2.imwrite('img/results/piano.jpg',img)

        cap.release()
        return img
    
    def fix_key_coords(black_keys,white_keys,shape):
        #adjust the coordinates of black and white keys to the video
        new_b_keys = [(i[0]+shape[0], i[1]+shape[1], i[2], i[3]) for i in black_keys]
        new_w_keys = [(i[0]+shape[0], i[1]+shape[1], i[2], i[3]) for i in white_keys]
        
        return(new_b_keys,new_w_keys)
    
    def get_key(image,black_keys,white_keys):
        #extract black key images from the input image based on the coordinates
        black_imgs = []
        for i in black_keys:
            new_image = []
            for row in range(i[0],i[0]+i[2]):
                new_image.append([])
                for col in  range(i[1],i[1]+i[3]):
                    new_image[row-i[0]].append(image[col][row])

            new_image = np.array(new_image)
            black_imgs.append(new_image)

        #extract white key images from the input image
        white_imgs = []
        for i in white_keys:
            new_image = []
            for row in range(i[0],i[0]+i[2]):
                new_image.append([])
                for col in  range(i[1],i[1]+i[3]):
                    new_image[row-i[0]].append(image[col][row])
            new_image = np.array(new_image)
            white_imgs.append(new_image)

        return(black_imgs,white_imgs)



        
       