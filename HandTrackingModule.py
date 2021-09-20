#Creating Module File to Use HandTracker for Multiple Projects and accessing point Dynamically 
import cv2
import mediapipe as mp
import time


#Creating Class HandDetector
class handDetector():
    
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands= maxHands
        self.detectionConfidence=detectionConfidence
        self.trackConfidence=trackConfidence
             
        # Hand Tracking
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionConfidence,self.trackConfidence) #Parameters - (Only Detection/Both: True/Flase, No. of Hands, min confidence detection, min confidence tracking)
        self.mpDraw = mp.solutions.drawing_utils

    #Funtion to Detect Hands and Landmarks
    def findHands(self,img, draw=True):
        #Converting to RGB
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        #Processing Image 
        self.results = self.hands.process(imgRGB)


        #Loop to Validate number of Hands
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    #Drawing Points and Lines on Tracked Hands
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    #Funtion to find Location of Each Point in the Video
    def findPosition(self, img, handNo=0,draw=True):
    
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                #print(id,lm)
                #Finding Pixel Location of Points
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])
                #Marking the Point 0 in Hand Landmarks
                if draw: 
                    cv2.circle(img, (cx,cy),25, (255,0,255), cv2.FILLED)
        return lmList

def main():
    
    cap = cv2.VideoCapture(0)
    
    #Calculating FPS
    #Previous Time
    pTime = 0
    #Current Time
    cTime = 0

    detector = handDetector()
    
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        
        lmList= detector.findPosition(img)
        
        if len(lmList) !=0:
            print(lmList[4]) #Thumb Position at 4
        
        #FPS Calculation and Printing
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255), 3)


        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()