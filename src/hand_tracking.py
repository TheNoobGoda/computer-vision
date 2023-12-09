import cv2
import mediapipe as mp
import numpy as np
from skimage.metrics import structural_similarity


class HandTrack:
    def handTrakc(src_vid_path,black_keys,white_keys,black_imgs,white_imgs,see_video):
        cap = cv2.VideoCapture(src_vid_path)

        mpHands = mp.solutions.hands
        hands = mpHands.Hands()
        keys = []
        last_key = []
        successe, img = cap.read()
        while successe:
            
            #remove this code after
            img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
            #up to here

            show_img = img.copy()

            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            #get hand landmarks
            results = hands.process(imgRGB)

            finger_coords = []
            if results.multi_hand_landmarks:
                #extract hand landmarks and finger coordinates
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, _ = img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)

                        #consider only the landmarks for the finger tips
                        if id in [4,8,12,16,20]:
                            cv2.circle(show_img,(cx,cy), 15, (255,0,255),cv2.FILLED)
                            finger_coords.append((cx,cy))

            key = []
            if len(finger_coords) != 0:
                for finger in finger_coords:
                    index = 0
                    for i in black_keys[:]:
                        #check if the finger is on a black key
                        if ( finger[0]> i[0] and finger[0] < i[0]+i[2] and finger[1] > i[1] and finger[1] < i[1]+i[3]):
                            #extract the region around the detected key
                            new_image = []
                            for row in range(i[0],i[0]+i[2]):
                                new_image.append([])
                                for col in  range(i[1],i[1]+i[3]):
                                    new_image[row-i[0]].append(img[col][row])

                            new_image = np.array(new_image)

                            #remove fingers from image
                            y = finger[1]-i[1]
                            for finger2 in finger_coords:
                                if finger2[1] < finger[1] and finger2[0]> i[0] and finger2[0] < i[0]+i[2] and finger2[1] > i[1] and finger2[1] < i[1]+i[3]:
                                    y = finger2[1]-i[1]
                            if y > 30 : y -=30
                            else: y = 0
                            if y != 0:
                                new_image = new_image[:,:y]
                                new_key_img = black_imgs[index][:,:y]

                                #convert images to grayscale for structural similarity comparison
                                gray_image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
                                gray_image2 = cv2.cvtColor(new_key_img, cv2.COLOR_BGR2GRAY)

                                x1,y1 = gray_image1.shape
                                x2,y2 = gray_image2.shape
                                ssim = None

                                #check if the structural similarity is below a threshold
                                if x1 >= 7 and x2 >= 7 and y1 >=7 and y2 >=7:
                                    ssim,_ = structural_similarity(gray_image1,gray_image2, full=True)
                                    # if ssim < 0.9:
                                    #     cv2.rectangle(show_img,(i[0],i[1]),(i[0]+i[2],i[3]+i[1]),(255,0,0),1)
                                    #     if ('b',i) not in key:
                                    #         key.append(('b',i))
                        index +=1
                    
                    index = 0
                    for i in white_keys[:]:
                        #check if the finger is on a white key
                        if ( finger[0]> i[0] and finger[0] < i[0]+i[2] and finger[1] > i[1] and finger[1] < i[1]+i[3]):
                            #extract the region around the detected key
                            new_image = []
                            for row in range(i[0],i[0]+i[2]):
                                new_image.append([])
                                for col in  range(i[1],i[1]+i[3]):
                                    new_image[row-i[0]].append(img[col][row])

                            new_image = np.array(new_image)

                            #remove fingers from image
                            y = finger[1]-i[1]
                            if y > 20 : y -=20
                            new_image = new_image[:,:y]
                            new_key_img = white_imgs[index][:,:y]

                            #convert images to grayscale for structural similarity comparison
                            gray_image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
                            gray_image2 = cv2.cvtColor(new_key_img, cv2.COLOR_BGR2GRAY)

                            x1,y1 = gray_image1.shape
                            x2,y2 = gray_image2.shape
                            ssim = None

                            #check if the structural similarity is below a threshold
                            if x1 >= 7 and x2 >= 7 and y1 >=7 and y2 >=7:
                                ssim,_ = structural_similarity(gray_image1,gray_image2, full=True)
                                if ssim < 0.65:
                                    cv2.rectangle(show_img,(i[0],i[1]),(i[0]+i[2],i[3]+i[1]),(255,0,0),1)
                                    if ('w',i) not in key:
                                        key.append(('w',i))
                        index +=1
                
            #see video
            if see_video:
                cv2.imshow("Image", show_img)
                cv2.waitKey(1)
            
            #read the next frame
            successe, img = cap.read()

            #store detected key in the keys list
            if key !=[] and key != last_key: 
                keys.append(key)
                last_key = key

        return(keys)