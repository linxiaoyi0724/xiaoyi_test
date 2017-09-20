# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:41:48 2017

@author: xiaoyi
"""
import caffe
import pandas as pd
import numpy as np

path = '/home/xiaoyi/data/MNIST/'
test_path = path+'test_kaggle/'

#train 
'''
solver_prototxt = path+'MNIST_lenet_kaggle/lenet_solver.prototxt'
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver(solver_prototxt)
solver.solve()
'''

#test
labels_filename = path+'labels.txt'
deply = path+'MNIST_lenet_kaggle/lenet.prototxt'
caffe_model = path + 'MNIST_kaggle_model/kaggle__iter_10000.caffemodel'
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(deply,caffe_model,caffe.TEST)

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1))
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data',(2,1,0))

labels = np.loadtxt(labels_filename,str,delimiter = '\t')
a = []
n = 28000
for i in xrange(n):
    im = caffe.io.load_image(test_path+str(i)+'.jpg')
    net.blobs['data'].data[...] = transformer.preprocess('data',im)
    net.forward()
    prob = net.blobs['prob'].data[0].flatten()
    order = prob.argsort()[-1]
    a.append(labels[order])
ImageId = [i+1 for i in range(n)]
submit_pd = pd.DataFrame({'ImageId':ImageId,'Label':a})
submit_pd.to_csv(path+'model.csv',index=False)
#im = caffe.io.load_image(test_path+'2.jpg')
#net.blobs['data'].data[...] = transformer.preprocess('data',im)
#out = net.forward()
#prob = net.blobs['prob'].data[0].flatten()
#print prob
#order = prob.argsort()[-1]
#print 'the class is:',labels[order]














