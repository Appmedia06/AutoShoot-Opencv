import cv2
import time
from datetime import datetime
import faceDetectionModule as fdm
import HandTrackingModule as htm

###########################
cap = cv2.VideoCapture(0)
###########################
pTime = 0
tipIds = [4, 8, 12, 16, 20]
imgNum = 1
###########################
# 計數變數
camSource = -1
running = True
saveCount = 1
nSecond = 0
totalSec = 3
strSec = '321'
keyPressTime = 0.0
startTime = 0.0
timeElapsed = 0.0
startCounter = False
endCounter = False
haveImg = False

frameWidth = 640
frameHeight = 480
#############################
# 手部，臉部辨識模組
faceDetector = fdm.FaceDetector(minDetectionCon=0.65)
handDetector = htm.handDector()


while True:
    success, img = cap.read()

    # 找臉部
    img, facebbox = faceDetector.findFaces(img, drawConfident = False)
    #找手部
    img = handDetector.findHand(img, draw = False)
    lmList, handbbox = handDetector.findPosition(img, draw = False)

    
    if len(lmList) != 0:
        fingers = []


        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:  
            fingers.append(1)
        else:
            fingers.append(0)

        
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]: 
                fingers.append(1)
            else:
                fingers.append(0)


        # 倒數三秒，並拍照
        if startCounter:
            if nSecond < totalSec: 

                cv2.putText(img, strSec[nSecond], (500, 150), cv2.FONT_HERSHEY_DUPLEX, 
                            6, (255,255,255), 5)

                timeElapsed = (datetime.now() - startTime).total_seconds()


                if timeElapsed >= 1:
                    nSecond += 1

                    timeElapsed = 0
                    startTime = datetime.now()
            # 把最新拍的照片顯示出來
            else:
                if haveImg:
                    cv2.destroyWindow(str(saveCount-1) + '.jpg')
                cv2.imwrite('C:/Users/user/PycharmProjects/faceShoot/img/' + str(saveCount) + '.jpg', img)  
                fileImg = cv2.imread('C:/Users/user/PycharmProjects/faceShoot/img/' + str(saveCount) + '.jpg')
                cv2.imshow(str(saveCount) + '.jpg', fileImg)
                saveCount += 1
                startCounter = False
                nSecond = 1
                haveImg = True

        # 若升起五支手指頭，則會拍照
        if fingers.count(1) == 5:
            startCounter = True
            startTime = datetime.now()
            keyPressTime = datetime.now()


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS : {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 2)
    if success:
        cv2.imshow("Image", img)
    else:
        print('Camera did not provide frame.')
        break
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



