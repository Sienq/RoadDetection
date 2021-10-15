import cv2
import numpy as np
import tkinter as tk




class LaneDetector():
    def __init__(self,frame):
        self.frame = frame
    
    def preprocess_frame(self,threshold1=100,threshold2=255,kernelSize = (3,3)):
        self.grayframe = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        self.grayframe = cv2.addWeighted(self.grayframe,1,np.zeros(self.grayframe.shape,self.grayframe.dtype),20,-20)
        cv2.imshow('gray-dark',self.grayframe)
        self.grayframe = cv2.GaussianBlur(self.frame,ksize=(3,3),sigmaX=0)
        self.filteredFrame = cv2.Canny(self.grayframe,threshold1,threshold2,kernelSize)


    def getPerspective(self):
        mask = np.zeros_like(self.filteredFrame)
        h,w = mask.shape
        center = (int(w/2),int(h/2))
        cords = np.array([[[center[0]+150,center[1]-20],
                           [center[0]-100,center[1]-20],
                           [0+50,h-225],
                           [w-50,h-225]]])
        mask = cv2.fillPoly(mask,cords,(255,255,255))
        self.maskedImg = cv2.bitwise_and(self.filteredFrame,mask)
        cv2.imshow('maskedImg',self.maskedImg)
        masksize = np.array([[0,0],[400,0],[0,600],[400,600]],np.float32)
        cords = np.squeeze(cords).astype(np.float32)
        cords = cords[[1,0,2,3]]
        perspective= cv2.getPerspectiveTransform(cords,masksize)
        self.perspectiveImg= cv2.warpPerspective(self.filteredFrame,perspective,(400,600))


    def detectLanes(self):
        lines = cv2.HoughLinesP(self.perspectiveImg, 2, np.pi/180,300,minLineLength=40, maxLineGap=300)
        if lines is not None:

            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(self.perspectiveImg, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow('perspective',self.perspectiveImg)







if __name__ == '__main__':
    video = cv2.VideoCapture('dashcamshort.mp4')

    while True:
        ret,cap = video.read()
        
        capfordetection = LaneDetector(cap)
        capfordetection.preprocess_frame()
        capfordetection.mask()
        cv2.imshow('cam',capfordetection.frame)
        cv2.imshow('canny',capfordetection.filteredFrame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    video.release()
#(540,360) (750,360) (100,719) (1180,719)