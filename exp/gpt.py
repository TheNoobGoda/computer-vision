import cv2
import mediapipe as mp

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize OpenCV
cap = cv2.VideoCapture(0)  # Use the default camera (change to a file path for video)

while cap.isOpened():
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform hand tracking
    results = hands.process(frame_rgb)

    # Check if hands were detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for i, landmark in enumerate(landmarks.landmark):
                # Convert normalized coordinates to pixel coordinates
                h, w, c = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Display the frame with landmarks
    cv2.imshow("Hand Tracking", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()