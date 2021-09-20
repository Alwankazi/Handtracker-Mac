import numpy as np
import cv2
import time
import HandTrackingModule as htm
import math
import applescript

#Cam Size
wCam, hCam = 640,480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
# # Hand Tracking
# mpHands = mp.solutions.hands
# hands = mpHands.Hands() #Parameters - (Only Detection/Both: True/Flase, No. of Hands, min confidence detection, min confidence tracking)
# mpDraw = mp.solutions.drawing_utils

#Calculating FPS
#Previous Time
pTime = 0
#Current Time
cTime = 0

#Using Module by creating Object of a class Detector
detector = htm.handDetector(detectionConfidence=0.7)


while True:
    success, img = cap.read()
    
    #Find Hands Function
    img=detector.findHands(img)
    lmList=detector.findPosition(img, draw=False)
    if len(lmList) !=0:
        #8 for tip of Index and 4 for Tip of Thumb
#         print(lmList[4],lmList[8])
        
        #X and Y Pixel Location of 4 and 8
        x1,y1 = lmList[4][1],lmList[4][2] #[4, 800, 383] 2 and 3 are x1 and y1
        x2,y2 = lmList[8][1],lmList[8][2] #[8, 793, 359] 2 and 3 are x2 and y2
        #Center Point
        cx,cy = (x1+x2)//2, (y1+y2)//2
        
        #Marking the selected Points 4 and 8
        cv2.circle(img,(x1,y1), 15, (255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2), 15, (255,0,255),cv2.FILLED)
        cv2.line(img, (x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img,(cx,cy), 15, (255,0,255),cv2.FILLED)
        
        length = math.hypot(x2-x1,y2-y1)
        # print(length)

        #Hand Range - 15 - 300
        #Volume Range - 0 - 100
        vol = np.interp(length,[15,300],[0, 100]) #Using Numpy to convert Value to Volume Range
        # print(vol)


        applescript.AppleScript("set volume output volume "+ str(vol)).run()

        if length<50:
            cv2.circle(img,(cx,cy), 15, (0,255,0),cv2.FILLED)
        
    
    #FPS Calculation and Printing
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255), 3)
    

    cv2.imshow("Image",img)
    cv2.waitKey(1)