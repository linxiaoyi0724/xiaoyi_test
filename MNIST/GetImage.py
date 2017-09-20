# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 16:58:09 2017

@author: xiaoyi
"""

import pandas as pd 
import numpy as np
from PIL import Image
path = '/home/xiaoyi/data/MNIST/'
path1 = path+'train_kaggle/'
path2 = path+'val_kaggle/'
#train data 

train = pd.read_csv(path+'train.csv')
train_array = train.values
train_data = train_array[:,1:]
labels = train_array[:,0]
n = np.shape(train_array)[0]
f = open(path1+'/train.txt','w')
f1 = open(path2+'/val.txt','w')
for i in xrange(n):
    img=train_data[i,:].reshape(28,28)
    img = np.uint8(img)
    img1 = Image.fromarray(img)
    if i<32000:
        img1.save(path1+str(i)+'_'+str(labels[i])+'.jpg')
        f.write(path1+str(i)+'_'+str(labels[i])+'.jpg ')
        f.write(str(labels[i])+'\n')
    else:
        img1.save(path2+str(i)+'_'+str(labels[i])+'.jpg')
        f1.write(path2+str(i)+'_'+str(labels[i])+'.jpg ')
        f1.write(str(labels[i])+'\n')
f.close()
f1.close()

#test data
'''
test = pd.read_csv(path+'test.csv')
test_data = test.values
n = np.shape(test_data)[0]
for i in xrange(n):
    img = test_data[i,:].reshape(28,28)
    img = np.uint8(img)
    img1 = Image.fromarray(img)
    img1.save(path+'test_kaggle/'+str(i)+'.jpg')
'''



