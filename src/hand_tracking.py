import cv2
import mediapipe as mp
import time
import numpy as np


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
            img = cv2.rotate(img,cv2.ROTATE_180)
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
                            cv2.circle(img,(cx,cy), 15, (255,0,255),cv2.FILLED)
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
                            print(img.shape)
                            print(i)
                            for row in range(0,i[2]):
                                for col in range(0,i[3]):
                                    if black_imgs[index][row][col].all() != img[i[2]+row][i[0]+col].all():
                                        a = 1
                            cv2.rectangle(img,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(0,0,255),1)
                            key.append(("b",i))
                            #print(finger[2])
                        index +=1
                        # else :
                        #     cv2.rectangle(img,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(255,0,0),1)
                    for i in white_keys:
                        if ( finger[0]> i[0] and finger[0] < i[0]+i[2] and finger[1] > i[1] and finger[1] < i[1]+i[3]):
                            cv2.rectangle(img,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(0,0,255),1)
                            key.append(("w",i))
                            #print(finger[2])
                        # else :
                        #     cv2.rectangle(img,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(255,0,0),1)
                
            #cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

            #print(keys1,keys2)
            # cv2.imshow("Image", img)
            # cv2.waitKey(1)
            successe, img = cap.read()
            if key != [] and key != last_key: keys.append(key)
            last_key = key

        return(keys)