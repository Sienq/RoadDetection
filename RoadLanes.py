import cv2
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from collections import deque

METER_PER_PIXEL_Y = 30/720
METER_PER_PIXEL_X = 3.7/700
#! WSZEDZIE DAC FRAME
#! Naprawic video feed

class Line():
    def __init__(self,maxSamples=4):
        self.maxSamples = maxSamples
        self.recentXFit = deque(maxlen=self.maxSamples)
        self.currentFit = [np.array([False])]
        self.bestFit = None
        self.bestx = None
        self.detected = False
        self.radiusOfCurvature = None
        self.lineBasePos = None

    def updateLines(self,ally,allx):
        self.bestx = np.mean(allx,axis=0)
        newFit = np.polyfit(ally,allx,2)
        self.currentFit = newFit
        self.recentXFit.append(self.currentFit)
        fitCurve = np.polyfit(ally*METER_PER_PIXEL_Y,allx*METER_PER_PIXEL_X,2)
        evaluateY = np.max(ally)
        self.radiusOfCurvature = ((1+(2*fitCurve[0]*evaluateY*METER_PER_PIXEL_Y+ fitCurve[1])**2)**1.5) // np.absolute(2*fitCurve[0])

class LaneDetector():
    def __init__(self,frame = None,maxSamples=4):

        self.frame = frame

    #!Convert img to HLS and get only Light channel
    def get_frame(self,frame):
        self.frame = frame

    def preprocess_frame(self,threshold1=255/3,threshold2=255,kernelSize = (5,5)):

        self.frame = cv2.GaussianBlur(self.frame,kernelSize,sigmaX=0)
        self.hlsImg = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HLS).astype(float)     # Convert from BGR to H-HUE L-LIGHT S-Saturation format
        self.hlsLight = self.hlsImg[:,:,1]      # Get only L channel

    #!Create mask over ROI where road lane appears,
    def get_perspective(self,frame):

        mask = np.zeros_like(frame)
        h,w= mask.shape
        center = (int(w/2),int(h/2))    # center of video frame
        cords = np.array([[[center[0]+150,center[1]-20],
                           [center[0]-100,center[1]-20],
                           [0+220,h-225],
                           [w-220,h-225]]])      # coordinates of rectangle where road lanes are
        mask = cv2.fillPoly(mask,cords,(255,255,255)) 
        self.maskedImg = cv2.bitwise_and(frame,mask)    # bitwise operation to make all elements except mask black

        masksize = np.array([[0,0],[1024,0],[0,600],[1024,600]],np.float32)     #size of mask in "bird eye" view
        cords = np.squeeze(cords).astype(np.float32)
        cords = cords[[1,0,2,3]]    # rearrange cords for get "getperspectivetransform" function
        perspective= cv2.getPerspectiveTransform(cords,masksize)    # get perspective from original image to "bird eye" view
        self.perspectiveImg= cv2.warpPerspective(frame,perspective,(1024,600),flags=cv2.INTER_LANCZOS4)     # extract "bird eye" image

    #!Prepare frame for making histogram
    def get_img_for_histogram(self):

       edge = cv2.Scharr(self.perspectiveImg,cv2.CV_64F,1,0)    # apply Schaar filter for edge detection
       edge = np.absolute(edge)     # we don't care if it's positive or negative
       edge = np.uint8(255* edge / np.max(edge))    # convert to 0-255 scale
       cv2.imshow('edges',edge)

       return edge

    def threshold(self):

        frame = self.get_img_for_histogram() # get frame afer filtering with Scharr
        self.binaryOutput = np.zeros_like(frame)

        height = self.binaryOutput.shape[0]
        thresholdChangeUp = 15
        thresholdChangeDown = 60
        thresholdChangeDelta = thresholdChangeDown-thresholdChangeUp

        for yindx in range(height):     # change threshold gradually to avoid disappearing line on top side of frame
            binLine = self.binaryOutput[yindx,:]
            edgeLine = frame[yindx,:]
            thresholdLine = thresholdChangeUp + thresholdChangeDelta * yindx / height
            binLine[edgeLine >= thresholdLine] = 255


        self.binaryOutput[(self.perspectiveImg >=140) & (self.perspectiveImg <= 255)] = 255
        cv2.imshow('binary',self.binaryOutput)


    #!Create histograms from tresholded "bird eye" view
    #!DONE
    def create_histograms(self,frame=None):

        self.partialFrame = self.binaryOutput # get small part of whole masked area
        cv2.imshow('partial',self.partialFrame)
        histogram = np.sum(self.partialFrame,axis=0)    # create histogram
        # plt.plot(histogram)
        # plt.show()

        leftLane = np.argmax(histogram[0:len(histogram)//2])    # left side of histogram is for left lane
        rightLane = np.argmax(histogram[len(histogram)//2:]) + int(histogram.shape[0]/2)   # right side of histogram is for right lane

        return leftLane,rightLane
    
    #!DONE
    def full_window_search(self,frame=None):

        leftLaneHist,rightLaneHist = self.create_histograms()

        slidingWindowsNumber = 10
        slidingWindowHeight = self.partialFrame.shape[0]/slidingWindowsNumber

        nonZero = self.partialFrame.nonzero()
        nonZeroY = np.array(nonZero[0])
        nonZeroX = np.array(nonZero[1])

        leftCurr = leftLaneHist
        rightCurr = rightLaneHist

        margin = 100
        minpix = 10

        self.leftIndicies = []
        self.rightIndicies = []
        self.toshow = np.array(self.partialFrame)
        for window in range(slidingWindowsNumber):
            winYLOW = np.round(self.partialFrame.shape[0] - (window+1) * slidingWindowHeight).astype(int)
            winYHIGH = np.round(self.partialFrame.shape[0] - window*slidingWindowHeight).astype(int)
            winXLeftLow = np.round(leftCurr - margin).astype(int)
            winXLeftHigh = np.round(leftCurr + margin).astype(int)
            winXRightLow = np.round(rightCurr - margin).astype(int)
            winXRightHigh = np.round(rightCurr + margin).astype(int)


            cv2.rectangle(self.toshow,(winXLeftLow,winYLOW),(winXLeftHigh,winYHIGH),(255,255,255),2)
            cv2.rectangle(self.toshow,(winXRightLow,winYLOW),(winXRightHigh,winYHIGH),(255,255,255),2)

            goodLeft = ((nonZeroY >= winYLOW) & (nonZeroY < winYHIGH) & (nonZeroX >= winXLeftLow) & (nonZeroX < winXLeftHigh)).nonzero()[0]
            goodRight = ((nonZeroY >= winYLOW) & (nonZeroY < winYHIGH) & (nonZeroX >= winXRightLow) & (nonZeroX < winXRightHigh)).nonzero()[0]

            self.leftIndicies.append(goodLeft)
            self.rightIndicies.append(goodRight)
            if len(goodLeft) > minpix:
                leftCurr = int(np.mean(nonZeroX[goodLeft]))
            if len(goodRight) > minpix:
                rightCurr = int(np.mean(nonZeroX[goodRight]))

        
        self.leftIndicies = np.concatenate(self.leftIndicies)
        self.rightIndicies = np.concatenate(self.rightIndicies)


        leftX = nonZeroX[self.leftIndicies]
        leftY = nonZeroY[self.leftIndicies]
        rightX = nonZeroX[self.rightIndicies]
        rightY = nonZeroY[self.rightIndicies]

        # self.partialFrame = cv2.cvtColor(self.partialFrame,cv2.COLOR_GRAY2BGR)
        ploty = np.linspace(0,self.partialFrame.shape[0]-1,self.partialFrame.shape[0])

        if len(leftX) and len(leftY):
            leftPolynomialFit = np.polyfit(leftY,leftX,2)
            leftFit = leftPolynomialFit[0] *ploty**2 + leftPolynomialFit[1]*ploty + leftPolynomialFit[2]
            left = np.asarray(tuple(zip(leftFit,ploty)),np.int32)
            cv2.polylines(self.toshow,[left],False,(255,255,255),10)

        if len(rightX) and len(rightY):
            rightPolynomialFit = np.polyfit(rightY,rightX,2)
            rightFit = rightPolynomialFit[0] *ploty**2 + rightPolynomialFit[1]*ploty + rightPolynomialFit[2]
            right = np.asarray(tuple(zip(rightFit,ploty)),np.int32)
            cv2.polylines(self.toshow,[right],False,(255,255,255),10)

        cv2.imshow('cam',self.toshow)



    def margin_search(self,leftLine,rightLine):
        print(leftLine.shape)
        print(rightLine.shape)
        nonZero = self.partialFrame.nonzero() # returns indexes of non zero pixels
        nonZeroY = np.array(nonZero[0]) # in Y direction
        nonZeroX = np.array(nonZero[1]) # in X direction

        margin = 30
        self.leftIndicies = ((nonZeroX > (leftLine[0]*(nonZeroY**2) + leftLine[1]*nonZeroY + leftLine[2] - margin)) & (nonZeroX < (leftLine[0]*(nonZeroY**2) + leftLine[1]*nonZeroY + leftLine[2] + margin)))
        self.rightIndicies = ((nonZeroX > (rightLine[0]*(nonZeroY**2) + rightLine[1]*nonZeroY + rightLine[2] - margin)) & (nonZeroX < (rightLine[0]*(nonZeroY**2) + rightLine[1]*nonZeroY + rightLine[2] + margin)))

        leftX = nonZeroX[self.leftIndicies]
        leftY = nonZeroY[self.leftIndicies]
        rightX = nonZeroX[self.rightIndicies]
        rightY = nonZeroY[self.rightIndicies]

        self.toshow = np.array(self.partialFrame)

        ploty = np.linspace(0,self.partialFrame.shape[0]-1,self.partialFrame.shape[0])

        if len(leftX) and len(leftY):
            leftPolynomialFit = np.polyfit(leftY,leftX,2)
            leftFit = leftPolynomialFit[0] * ploty**2 + leftPolynomialFit[1]*ploty + leftPolynomialFit[2]
            leftLWindow1 = np.array([np.transpose(np.vstack([leftFit-margin,ploty]))])
            leftLWindow2 = np.array([np.flipud(np.transpose(np.vstack([leftFit+margin, ploty])))])
            leftLinePoints = np.hstack((leftLWindow1,leftLWindow2))
            cv2.fillPoly(self.toshow,np.intc([leftLinePoints]),(255,255,255))
            left = np.asarray(tuple(zip(leftFit,ploty)),np.int32)
            cv2.polylines(self.toshow,[left],False,(255,255,255),3)
        if len(rightX) and len(rightY):
            rightPolynomialFit = np.polyfit(rightY,rightX,2)
            rightFit = rightPolynomialFit[0] *ploty**2 + rightPolynomialFit[1]*ploty + rightPolynomialFit[2]
            rightLWindow1 = np.array([np.transpose(np.vstack([rightFit-margin,ploty]))])
            rightLWindow2 = np.array([np.flipud(np.transpose(np.vstack([rightFit+margin, ploty])))])
            rightLinePoints = np.hstack((rightLWindow1,rightLWindow2))
            cv2.fillPoly(self.toshow,np.intc([rightLinePoints]),(255,255,255))
            right = np.asarray(tuple(zip(rightFit,ploty)),np.int32)
            cv2.polylines(self.toshow,[right],False,(255,255,255),3)

        cv2.imshow('cam',self.toshow)





    def validate_find_lanes(self,frame,leftLine,rightLine):

        imageSizes = (self.partialFrame.shape[1],frame.shape[0])

        nonzero = self.partialFrame.nonzero()
        nonZeroY = np.array(nonzero[0])
        nonZeroX = np.array(nonzero[1])

        leftLineAllX = nonZeroX[self.leftIndicies]
        leftLineAllY = nonZeroY[self.leftIndicies]
        rightLineAllX = nonZeroX[self.rightIndicies]
        rightLineAllY = nonZeroY[self.rightIndicies]

        if len(leftLineAllX) < 1800 or len(rightLineAllX) < 1800:
            print('0 if')
            leftLine.detected = False
            rightLine.detected = False
            return

        leftXMean = np.mean(leftLineAllX,axis=0)
        rightXMean = np.mean(rightLineAllX,axis=0)
        print(leftXMean,rightXMean)
        laneWidth = np.subtract(rightXMean,leftXMean)

        if leftXMean > 512 or rightXMean < 512:
            print(leftXMean,rightXMean,'first if')
            leftLine.detected = False
            rightLine.detected = False
            return

        if laneWidth < 300 or laneWidth > 600:
            # print('second if')
            leftLine.detected = False
            rightLine.detected = False
            return

        if leftLine.bestx is None or np.abs(np.subtract(leftLine.bestx,np.mean(leftLineAllX,axis=0))) < 100:
            print('3rd if')
            leftLine.updateLines(leftLineAllY,leftLineAllX)
            leftLine.detected = True
            
        else:
            # print('1st else')
            leftLine.detected = False

        if rightLine.bestx is None or np.abs(np.subtract(rightLine.bestx,np.mean(rightLineAllX,axis=0))) < 100:
            print('4th if')
            rightLine.updateLines(rightLineAllY,rightLineAllX)
            rightLine.detected = True
        
        else:
            # print('2nd else')
            rightLine.detected = False

        carPos = imageSizes[0]/2
        leftFit = leftLine.currentFit
        rightFit = rightLine.currentFit
        leftLaneBasePoly = leftFit[0]*imageSizes[1]**2 + leftFit[1]*imageSizes[1] + leftFit[2]
        rightLaneBasePoly = rightFit[0]*imageSizes[1]**2 + rightFit[1]*imageSizes[1] + rightFit[2]

        laneCenter = (leftLaneBasePoly + rightLaneBasePoly) / 2

        leftLine.lineBasePos = (carPos - laneCenter) * METER_PER_PIXEL_X
        rightLine.lineBasePos = leftLine.lineBasePos


        

    def find_lanes(self,leftLine,rightLine):
        if leftLine.detected and rightLine.detected:
            self.margin_search(leftLine.currentFit,rightLine.currentFit)
            self.validate_find_lanes(self.partialFrame,leftLine,rightLine)
            cv2.imshow('cam',self.toshow)
        else:
            self.full_window_search()
            self.validate_find_lanes(self.partialFrame,leftLine,rightLine)
            cv2.imshow('cam',self.toshow)

if __name__ == '__main__':
    video = cv2.VideoCapture('dashcamshort.mp4')
    leftLine = Line()
    rightLine = Line()
    capfordetection = LaneDetector()
    while True:
        ret,cap = video.read()

        capfordetection.get_frame(cap)
        capfordetection.preprocess_frame()
        capfordetection.get_perspective(capfordetection.hlsLight)
        capfordetection.get_img_for_histogram()
        capfordetection.threshold()
        capfordetection.find_lanes(leftLine,rightLine)
        cv2.imshow('cam',capfordetection.toshow)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    video.release()