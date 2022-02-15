import cv2
import mediapipe as mp
import time
import math

class handDector():
    def __init__(self, mode=False, maxHand=2, detectionCon=1, trackCon=0.5):
        self.mode = mode
        self.maxHand = maxHand
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHand,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.handLmsStyle = self.mpDraw.DrawingSpec(color=(0,0,255), thickness=5)
        self.handConStyle = self.mpDraw.DrawingSpec(color=(0,255,0), thickness=10)
        self.tipIds = [4, 8, 12, 16, 20]

    def findHand(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                                               self.handLmsStyle, self.handConStyle)
        return img
    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = [] #邊界
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 6, (0,0,255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax  
            if draw:
                cv2.rectangle(img, (xmin - 20, ymin -20), (xmax + 20, ymax + 20),
                            (0, 255, 0), 2)
        return self.lmList, bbox
    
    def fingersUp(self):
        fingers = []
        #Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Other fingers
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0 , 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0 , 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0 , 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1) #平方開根號(a,b)=(a**2+b**2)**0.5

        return length, img, [x1, y1, x2, y2, cx, cy]
        

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDector()
    while True:
        success, img = cap.read()
        img = detector.findHand(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS : {int(fps)}', (10,70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255,0,255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()