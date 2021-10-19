import cv2
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt


class LaneDetector():
    def __init__(self,frame):

        self.frame = frame

    #!Convert img to HLS and get only Light channel
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
    def create_histograms(self,frame=None):

        self.partialFrame = self.binaryOutput # get small part of whole masked area
        cv2.imshow('partial',self.partialFrame)
        histogram = np.sum(self.partialFrame,axis=0)    # create histogram
        # plt.plot(histogram)
        # plt.show()

        leftLane = np.argmax(histogram[0:len(histogram)//2])    # left side of histogram is for left lane
        rightLane = np.argmax(histogram[len(histogram)//2:]) + int(histogram.shape[0]/2)   # right side of histogram is for right lane

        return leftLane,rightLane
    #! JAK NIE ZADZIALA TO ZMIENIC PARTIAL FRAME NA BINARY OUTPUT

    def full_window_search(self,frame=None):

        leftLaneHist,rightLaneHist = self.create_histograms()
        
        slidingWindowsNumber = 7
        slidingWindowHeight = self.partialFrame.shape[0]/slidingWindowsNumber

        nonZero = self.partialFrame.nonzero()
        nonZeroY = np.array(nonZero[0])
        nonZeroX = np.array(nonZero[1])

        leftCurr = leftLaneHist
        rightCurr = rightLaneHist

        margin = 100
        minpix = 10

        leftIndicies = []
        rightIndicies = []

        for window in range(slidingWindowsNumber):
            winYLOW = np.round(self.partialFrame.shape[0] - (window+1) * slidingWindowHeight).astype(int)
            winYHIGH = np.round(self.partialFrame.shape[0] - window*slidingWindowHeight).astype(int)
            winXLeftLow = np.round(leftCurr - margin).astype(int)
            winXLeftHigh = np.round(leftCurr + margin).astype(int)
            winXRightLow = np.round(rightCurr - margin).astype(int)
            winXRightHigh = np.round(rightCurr + margin).astype(int)


            cv2.rectangle(self.partialFrame,(winXLeftLow,winYLOW),(winXLeftHigh,winYHIGH),(255,255,255),2)
            cv2.rectangle(self.partialFrame,(winXRightLow,winYLOW),(winXRightHigh,winYHIGH),(255,255,255),2)

            goodLeft = ((nonZeroY >= winYLOW) & (nonZeroY < winYHIGH) & (nonZeroX >= winXLeftLow) & (nonZeroX < winXLeftHigh)).nonzero()[0]
            goodRight = ((nonZeroY >= winYLOW) & (nonZeroY < winYHIGH) & (nonZeroX >= winXRightLow) & (nonZeroX < winXRightHigh)).nonzero()[0]

            leftIndicies.append(goodLeft)
            rightIndicies.append(goodRight)
            if len(goodLeft) > minpix:
                leftCurr = int(np.mean(nonZeroX[goodLeft]))
            if len(goodRight) > minpix:
                rightCurr = int(np.mean(nonZeroX[goodRight]))


        leftIndicies = np.concatenate(leftIndicies)
        rightIndicies = np.concatenate(rightIndicies)

        leftX = nonZeroX[leftIndicies]
        leftY = nonZeroY[leftIndicies]
        rightX = nonZeroX[rightIndicies]
        rightY = nonZeroY[rightIndicies]
        
        self.partialFrame = cv2.cvtColor(self.partialFrame,cv2.COLOR_GRAY2BGR)
        
        ploty = np.linspace(0,self.partialFrame.shape[0]-1,self.partialFrame.shape[0])
        
        
        if len(leftX) and len(leftY):
            leftPolynomialFit = np.polyfit(leftY,leftX,2)
            leftFit = leftPolynomialFit[0] *ploty**2 + leftPolynomialFit[1]*ploty + leftPolynomialFit[2]
            left = np.asarray(tuple(zip(leftFit,ploty)),np.int32)
            cv2.polylines(self.partialFrame,[left],False,(255,255,0),10)
        if len(rightX) and len(rightY):
            rightPolynomialFit = np.polyfit(rightY,rightX,2)
            rightFit = rightPolynomialFit[0] *ploty**2 + rightPolynomialFit[1]*ploty + rightPolynomialFit[2]
            right = np.asarray(tuple(zip(rightFit,ploty)),np.int32)
            cv2.polylines(self.partialFrame,[right],False,(255,0,255),10)

        cv2.imshow('with rect',self.partialFrame)



    def margin_search(self,frame = None):

        nonZero = self.partialFrame.nonzero()
        nonZeroY = np.array(nonZero[0])
        nonZeroX = np.array(nonZero[1])

        margin = 50

        leftIndicies = ((nonZeroX > (leftLine.current_fit[0]*(nonZeroY**2) + leftLine.current_fit[1]*nonZeroY + leftLine.current_fit[2] - margin)) & (nonZeroX < (leftLine.current_fit[0]*(nonZeroY**2) + leftLine.current_fit[1]*nonZeroY + leftLine.current_fit[2] + margin)))
        rightIndicies = ((nonZeroX > (rightLine.current_fit[0]*(nonZeroY**2) + rightLine.current_fit[1]*nonZeroY + rightLine.current_fit[2] - margin)) & (nonZeroX < (rightLine.current_fit[0]*(nonZeroY**2) + rightLine.current_fit[1]*nonZeroY + rightLine.current_fit[2] + margin)))

        leftX = nonZeroX[leftIndicies]
        leftY = nonZeroY[leftIndicies]
        rightX = nonZeroX[rightIndicies]
        rightY = nonZeroY[rightIndicies]

        self.partialFrame = cv2.cvtColor(self.partialFrame,cv2.COLOR_GRAY2BGR)
        
        ploty = np.linspace(0,self.partialFrame.shape[0]-1,self.partialFrame.shape[0])

        if len(leftX) and len(leftY):
            leftPolynomialFit = np.polyfit(leftY,leftX,2)
            leftFit = leftPolynomialFit[0] *ploty**2 + leftPolynomialFit[1]*ploty + leftPolynomialFit[2]
            left = np.asarray(tuple(zip(leftFit,ploty)),np.int32)
            cv2.polylines(self.partialFrame,[left],False,(255,255,0),10)
        if len(rightX) and len(rightY):
            rightPolynomialFit = np.polyfit(rightY,rightX,2)
            rightFit = rightPolynomialFit[0] *ploty**2 + rightPolynomialFit[1]*ploty + rightPolynomialFit[2]
            right = np.asarray(tuple(zip(rightFit,ploty)),np.int32)
            cv2.polylines(self.partialFrame,[right],False,(255,0,255),10)

if __name__ == '__main__':
    video = cv2.VideoCapture('dashcamshort.mp4')

    while True:
        ret,cap = video.read()

        capfordetection = LaneDetector(cap)
        capfordetection.preprocess_frame()
        capfordetection.get_perspective(capfordetection.hlsLight)
        capfordetection.get_img_for_histogram()
        capfordetection.threshold()
        capfordetection.full_window_search()
        cv2.imshow('cam',capfordetection.frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    video.release()