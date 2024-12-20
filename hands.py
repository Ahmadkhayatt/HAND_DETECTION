import cv2
import time    
import numpy as np
import mediapipe as mp
cap = cv2.VideoCapture(0)

face_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0 
cTime = 0
while True :

    success , frame = cap.read()
    imgRGB = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi.hand_landmarks)
    if  results.multi_hand_landmarks :
        for handLms in  results.multi_hand_landmarks :
            mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS)
    
    
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face_model.detectMultiScale(gray , 1.5 , 5)

    for (x,y,w,h )in faces :
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),5)
        roi_gray = gray[y:y+w,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
       


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(frame, str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3)

    cv2.imshow('frmae',frame)
    if cv2.waitKey(1) == ord ('q'):
        break 
cv2.destroyAllWindows()   
