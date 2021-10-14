import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk

if __name__=='__main__':

    root = tk.Tk()
    # root.geometry('1280x720')
    root.configure(bg='black')
    tk.Label(root,text='Detection',bg='white',fg='black').pack()
    f1 = tk.LabelFrame(root,bg='white')
    f1.pack()
    f2 = tk.LabelFrame(root,bg='white')
    f2.pack()
    vid = tk.Label(f1,bg='white')
    vid.pack()

    vid2 =tk.Label(f2,bg='white')
    vid2.pack()


    video = cv2.VideoCapture('test.mp4')

    while True:
        ret, img = video.read()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imggray = ImageTk.PhotoImage(Image.fromarray(imggray))
        img = ImageTk.PhotoImage(Image.fromarray(img))
        vid['image'] = img
        vid2['image'] = imggray
        root.update()




