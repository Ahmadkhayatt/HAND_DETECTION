import cv2
import time
import numpy as np
import mediapipe as mp

cap = cv2.VideoCapture(0)
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Set up face detection
face_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize FPS variables
pTime = 0
cTime = 0

# Start capturing video
while True:
    success, frame = cap.read()
    if not success:
        break
    
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Drawing hand landmarks
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            
            # Get list of all landmark coordinates
            landmarks = handLms.landmark
            
            # List of finger tip indexes
            finger_tips = [4, 8, 12, 16, 20]
            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            raised_fingers = []

            # Check each finger if it is raised
            for i, tip in enumerate(finger_tips):
                # For thumb, check if it's to the right/left of its joint
                if tip == 4:  # Thumb
                    if (landmarks[tip].x < landmarks[tip - 2].x and landmarks[tip].y < landmarks[tip - 2].y):
                        raised_fingers.append("Thumb")
                else:  # Other fingers (y-coordinate only)
                    if landmarks[tip].y < landmarks[tip - 2].y:  # Check if tip is higher than lower joint
                        raised_fingers.append(finger_names[i])

            # Display which fingers are raised on the frame
            cv2.putText(frame, ', '.join(raised_fingers) if raised_fingers else "No fingers raised",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_model.detectMultiScale(gray, 1.5, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    # Show the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
