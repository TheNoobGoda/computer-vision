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

        while True:
            success, img = cap.read()

            #remove this code after
            img = cv2.rotate(img,cv2.ROTATE_180)
            resize = []

            for i in range(img.shape[0]):
                resize.append([])
                for j in range(0,img.shape[1]-50):
                    resize[i].append(img[i][j])

            resize = np.array(resize) 
            #up to here

            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        #print(id,cx,cy)
                        if id in [4,8,12,16,20]:
                            cv2.circle(img,(cx,cy), 15, (255,0,255),cv2.FILLED)
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # cTime = time.time()
            # fps = 1/(cTime-pTime)
            # pTime = cTime

            for i in black_keys:
                cv2.rectangle(img,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(255,0,0),1)
            
            for i in white_keys:
                cv2.rectangle(img,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(0,255,0),1)
            
            #cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)


            cv2.imshow("Image", img)
            cv2.waitKey(1)