import cv2
import mediapipe as mp

class handDetector():
        
    def __init__(self, static_image_mode=False, max_num_hands=2, 
                 model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode, self.max_num_hands, self.model_complexity,
                                        self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self, frame):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        
        if self.results.multi_hand_landmarks: # results.multi_hand_landmarks give us x,y,z locations
            for i in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, i, self.mpHands.HAND_CONNECTIONS)
        return frame 

    def findPosition(self, frame, handNo=0):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand =self.results.multi_hand_landmarks[handNo]
            for i, lm, in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([i, cx, cy])
        return lmList

def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        ret, frame = cap.read()
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 13: break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
