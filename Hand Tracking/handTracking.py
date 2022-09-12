import cv2
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    
    
    if results.multi_hand_landmarks: # results.multi_hand_landmarks give us x,y,z locations
        for i in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, i, mpHands.HAND_CONNECTIONS)
           
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == 13: break

cap.release()
cv2.destroyAllWindows()
