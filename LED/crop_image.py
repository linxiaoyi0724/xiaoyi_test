# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 22:47:11 2017

@author: xiaoyi
"""

import cv2
import os

global img
global point1, point2,i
def on_mouse(event, x, y, flags, param):
    global img, point1, point2,i
    path1 = '/home/xiaoyi/data/LED/data/train/train_small_crop/'
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               #按住左键拖曳
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x,y)
        cv2.rectangle(img2, point1, point2, (0,0,255), 5) 
        cv2.imshow('image', img2)
        min_x = min(point1[0],point2[0])     
        min_y = min(point1[1],point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] -point2[1])
        cut_img = img[min_y:min_y+height, min_x:min_x+width]
        cv2.imwrite(path1+str(i)+'.jpg', cut_img)

def main():
    global img,i
    i =1
    path = '/home/xiaoyi/data/LED/data/train/train_small/'
    for files in os.listdir(path):
        img = cv2.imread(path+files)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',on_mouse)
        cv2.imshow('image',img)
        print(i)
        cv2.waitKey(0)
        i+=1
        
        
   

if __name__ == '__main__':
    main()
