import cv2
import time
import os
import handTrackingModule as htm

w, h = 640, 480
cap = cv2.VideoCapture(0)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
fingerList = []
for i in myList:
    image = cv2.imread(f"{folderPath}/{i}")
    fingerList.append(image)
  
detector = htm.handDetector(min_detection_confidence=0.75)  

tips = [4, 8, 12, 16, 20]

while True:
    ret, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame)
    
    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tips[0]][1] > lmList[tips[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Finger
        for i in range(1,5):
            if lmList[tips[i]][2] < lmList[tips[i]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        text = ""
        
        if fingers == [0,0,0,0,0]:  
            frame[0:128, 512:640] = fingerList[0]
            text = "0"              
        if fingers == [0,1,0,0,0]:
            frame[0:128, 512:640] = fingerList[1] 
            text = "1"
        if fingers == [0,1,1,0,0]:
            frame[0:128, 512:640] = fingerList[2] 
            text = "2"
        if fingers == [0,1,1,1,0]:
            frame[0:128, 512:640] = fingerList[3] 
            text = "3"            
        if fingers == [0,1,1,1,1]:
            frame[0:128, 512:640] = fingerList[4] 
            text = "4"      
        if fingers == [1,1,1,1,1]:
            frame[0:128, 512:640] = fingerList[5] 
            text = "5"
            
        cv2.rectangle(frame, (512,128), (640,246), (0,0,0), cv2.FILLED)
        cv2.putText(frame, text , (547,218), cv2.FONT_HERSHEY_COMPLEX, 3, (255,255,255), 10)
    
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) &0xFF == 13: break

cap.release()
cv2.destroyAllWindows()
