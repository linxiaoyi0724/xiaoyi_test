# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:13:37 2017

@author: xiaoyi
"""
import cv2
import os 


path = '/home/xiaoyi/data/LED/test/test_small1/'
path1 = '/home/xiaoyi/data/LED/test/test_small/'


i=1
for files in os.listdir(path):
    img = cv2.imread(path+files)
    cv2.imwrite(path1+str(i)+'.jpg',img)
    i+=1
    

#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()















#Rename Image

#i =1
#for files in os.listdir(path):
#    if os.path.isfile(os.path.join(path,files)) == True:
#        newname = '%s.bmp' %i
#        os.rename(os.path.join(path,files),os.path.join(path,newname))
#        i+=1
#        
    
    
    


    

