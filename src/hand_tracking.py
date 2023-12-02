import cv2
import mediapipe as mp
import time
import numpy as np


class HandTrack:
    def handTrakc(src_vid_path,black_keys,white_keys):
        cap = cv2.VideoCapture(src_vid_path)

        mpHands = mp.solutions.hands
        hands = mpHands.Hands()
        mpDraw = mp.solutions.drawing_utils

        pTime = 0
        cTime = 0
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
                        #print(id,cx,cy)
                        if id in [4,8,12,16,20]:
                            cv2.circle(img,(cx,cy), 15, (255,0,255),cv2.FILLED)
                            finger_coords.append((cx,cy))
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # cTime = time.time()
            # fps = 1/(cTime-pTime)
            # pTime = cTime
            keys1 = []
            keys2 = []
            if len(finger_coords) != 0:
                for finger in finger_coords:
                    for i in black_keys:
                        if ( finger[0]> i[0] and finger[0] < i[0]+i[2] and finger[1] > i[1] and finger[1] < i[1]+i[3]):
                            cv2.rectangle(img,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(0,0,255),1)
                            keys1.append(i)
                        # else :
                        #     cv2.rectangle(img,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(255,0,0),1)
                    
                    for i in white_keys:
                        if ( finger[0]> i[0] and finger[0] < i[0]+i[2] and finger[1] > i[1] and finger[1] < i[1]+i[3]):
                            cv2.rectangle(img,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(0,0,255),1)
                            keys2.append(i)
                        # else :
                        #     cv2.rectangle(img,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(255,0,0),1)
                
            #cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

            print(keys1,keys2)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
            successe, img = cap.read()