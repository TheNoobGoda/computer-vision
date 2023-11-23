import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                
                # Get the actual depth (z-coordinate) using the depth map
                depth_value = results.multi_hand_landmarks[0].landmark[id].z

                # Convert normalized depth to actual depth in millimeters
                actual_depth = int(depth_value * 1000)

                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(cx, cy, actual_depth)

            if id == 4 :
                cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
            cv2.putText(image, str(actual_depth), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

            cv2.imshow("Output", image)
            cv2.waitKey(1)