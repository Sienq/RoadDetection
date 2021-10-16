import cv2
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt


class LaneDetector():
    def __init__(self,frame):
        self.frame = frame

    def preprocess_frame(self,threshold1=100,threshold2=255,kernelSize = (3,3)):
        self.grayframe = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        self.grayframe = cv2.addWeighted(self.grayframe,0.02,np.zeros(self.grayframe.shape,self.grayframe.dtype),20,-20)
        cv2.imshow('gray',self.grayframe)

        self.grayframe = cv2.GaussianBlur(self.frame,ksize=(3,3),sigmaX=0)
        self.filteredFrame = cv2.Canny(self.grayframe,threshold1,threshold2,kernelSize)

        self.hlsImg = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HLS).astype(float)
        self.hlsLight = self.hlsImg[:,:,1]


    def get_perspective(self,frame):
        mask = np.zeros_like(frame)
        h,w= mask.shape
        center = (int(w/2),int(h/2))
        cords = np.array([[[center[0]+150,center[1]-20],
                           [center[0]-100,center[1]-20],
                           [0+50,h-225],
                           [w-50,h-225]]])
        mask = cv2.fillPoly(mask,cords,(255,255,255))
        self.maskedImg = cv2.bitwise_and(frame,mask)
        cv2.imshow('maskedImg',self.maskedImg)

        masksize = np.array([[0,0],[1024,0],[0,600],[1024,600]],np.float32)
        cords = np.squeeze(cords).astype(np.float32)
        cords = cords[[1,0,2,3]]
        perspective= cv2.getPerspectiveTransform(cords,masksize)
        self.perspectiveImg= cv2.warpPerspective(frame,perspective,(1024,600),flags=cv2.INTER_LANCZOS4)


    def get_img_for_histogram(self):

       edge = cv2.Scharr(self.perspectiveImg,cv2.CV_64F,1,0)
       edge = np.absolute(edge)
       edge = np.uint8(255* edge / np.max(edge))
       cv2.imshow('edges',edge)

       self.binaryOutputFrame = np.zeros_like(edge)
       self.binaryOutputFrame[edge >= 50] = 255
       cv2.imshow('binary',self.binaryOutputFrame)

    def detect_lines(self):
        partialFrame = self.binaryOutputFrame[self.binaryOutputFrame.shape[0] * 2 // 3:,:]
        cv2.imshow('partial',partialFrame)
        histogram = np.sum(partialFrame,axis=0)
        # plt.plot(histogram)
        # plt.show()

        leftLane = np.argmax(histogram[0:len(histogram)//2])
        rightLane = np.argmax(histogram[len(histogram)//2:])

        windowHeight = self.binaryOutputFrame.shape[0]/15 #num_windows
        margin = 80
        minPixels = 50
        nonZero = self.binaryOutputFrame.nonzero()
        nonZeroY = np.array(nonZero[0])
        nonzeroX = np.array(nonZero[1])

        leftX = leftLane
        rightX = rightLane

        leftLaneIndexes = []
        rightLaneIndexes = []

        for idx in range(15):
            windowXleftMin = leftX - margin
            windowXleftMax = leftX + margin
            windowXrightMin = leftX - margin
            windowXrightMax = leftX + margin

            windowYTop = partialFrame.shape[1] - idx * windowHeight
            windowYBottom = windowYTop - windowHeight

            cv2.imshow('lines',partialFrame)


if __name__ == '__main__':
    video = cv2.VideoCapture('dashcamshort.mp4')

    while True:
        ret,cap = video.read()

        capfordetection = LaneDetector(cap)
        capfordetection.preprocess_frame()
        capfordetection.get_perspective(capfordetection.hlsLight)
        capfordetection.get_img_for_histogram()
        capfordetection.detect_lines()
        cv2.imshow('cam',capfordetection.frame)
        cv2.imshow('canny',capfordetection.filteredFrame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    video.release()