import cv2
import mediapipe as mp
import time
import numpy as np
from skimage.metrics import structural_similarity


class HandTrack:
    def handTrakc(src_vid_path,black_keys,white_keys,black_imgs,white_imgs):
        cap = cv2.VideoCapture(src_vid_path)

        mpHands = mp.solutions.hands
        hands = mpHands.Hands()
        mpDraw = mp.solutions.drawing_utils

        pTime = 0
        cTime = 0
        keys = []
        last_key = []
        successe, img = cap.read()
        while successe:
            

            #remove this code after
            img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
            show_img = img.copy()
            #up to here

            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            finger_coords = []
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)

                        if id in [4,8,12,16,20]:
                        #if id == 8:
                            cv2.circle(show_img,(cx,cy), 15, (255,0,255),cv2.FILLED)
                            finger_coords.append((cx,cy))
                            #cv2.putText(img,str(actual_depth),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

            # cTime = time.time()
            # fps = 1/(cTime-pTime)
            # pTime = cTime
            key = []
            if len(finger_coords) != 0:
                for finger in finger_coords:
                    index = 0
                    for i in black_keys[:]:
                        if ( finger[0]> i[0] and finger[0] < i[0]+i[2] and finger[1] > i[1] and finger[1] < i[1]+i[3]):
                            new_image = []
                            for row in range(i[0],i[0]+i[2]):
                                new_image.append([])
                                for col in  range(i[1],i[1]+i[3]):
                                    new_image[row-i[0]].append(img[col][row])

                            new_image = np.array(new_image)
                            y = finger[1]-i[1]
                            for finger2 in finger_coords:
                                if finger2[1] < finger[1] and finger2[0]> i[0] and finger2[0] < i[0]+i[2] and finger2[1] > i[1] and finger2[1] < i[1]+i[3]:
                                    y = finger2[1]-i[1]
                            if y > 30 : y -=30
                            else: y = 0
                            if y != 0:
                                new_image = new_image[:,:y]
                                new_image = new_image[:,:y]
                                new_key_img = black_imgs[index][:,:y]
                                gray_image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
                                gray_image2 = cv2.cvtColor(new_key_img, cv2.COLOR_BGR2GRAY)
                                x1,y1 = gray_image1.shape
                                x2,y2 = gray_image2.shape
                                ssim = None
                                if x1 >= 7 and x2 >= 7 and y1 >=7 and y2 >=7:
                                    ssim,_ = structural_similarity(gray_image1,gray_image2, full=True)
                                    #print(f"black {index} {ssim}")
                                    if ssim < 0.7:
                                        cv2.rectangle(show_img,(i[0],i[1]),(i[0]+i[2],i[3]+i[1]),(255,0,0),1)
                                        key.append(('b',i))
                                #print(f"black key number {index}: {ssim}")
                        index +=1
                    index = 0
                    for i in white_keys[:]:
                        if ( finger[0]> i[0] and finger[0] < i[0]+i[2] and finger[1] > i[1] and finger[1] < i[1]+i[3]):
                            new_image = []
                            for row in range(i[0],i[0]+i[2]):
                                new_image.append([])
                                for col in  range(i[1],i[1]+i[3]):
                                    new_image[row-i[0]].append(img[col][row])

                            new_image = np.array(new_image)
                            y = finger[1]-i[1]
                            if y > 20 : y -=20
                            new_image = new_image[:,:y]
                            new_key_img = white_imgs[index][:,:y]
                            gray_image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
                            gray_image2 = cv2.cvtColor(new_key_img, cv2.COLOR_BGR2GRAY)
                            x1,y1 = gray_image1.shape
                            x2,y2 = gray_image2.shape
                            ssim = None
                            if x1 >= 7 and x2 >= 7 and y1 >=7 and y2 >=7:
                                ssim,_ = structural_similarity(gray_image1,gray_image2, full=True)
                                #print(f"white {index} {ssim}")
                                if ssim < 0.7:
                                    cv2.rectangle(show_img,(i[0],i[1]),(i[0]+i[2],i[3]+i[1]),(255,0,0),1)
                                    if ('w',i) not in key:
                                        key.append(('w',i))
                            #print(f"white key number {index}: {ssim}")
                        index +=1
                
            #cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

            #print(keys1,keys2)
            cv2.imshow("Image", show_img)
            cv2.waitKey(1)
            successe, img = cap.read()
            if key != [] and key != last_key: keys.append(key)
            last_key = key

        return(keys)