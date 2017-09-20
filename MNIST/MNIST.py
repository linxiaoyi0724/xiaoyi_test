# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 16:58:21 2017

@author: xiaoyi
"""

#==============================================================================
# #读取数据
# import numpy as np
# import pandas as pd
# import caffe
#==============================================================================
#==============================================================================
# def get_mnist_data(): 
#     #读取训练测试数据
#     train = pd.read_csv('train.csv')
#     test = pd.read_csv('test.csv')
#     sample_submission = pd.read_csv('sample_submission.csv')
#     train_np = np.array(train)
#     test_np = np.array(test)
#     train_data_totals = train_np[:,1:]
#     train_labels_totals =train_np[:,0] 
#     test_data = test_np[:,0:]
#     
#     #读取评价数据并打乱数据
#     num_train = len(train_labels_totals)
#     num_test  = np.size(test_data,axis=0)
#     num_val = num_train/10
#     num_train = num_train-num_val
#     
#     mask = range(num_train,num_train+num_val)
#     train_val_data = train_data_totals[mask]
#     train_val_labels = train_labels_totals[mask]
#     mask = range(num_train)
#     train_data = train_data_totals[mask]
#     train_labels = train_labels_totals[mask]
# 
# #求均值
#     mean_image = np.mean(train_data,axis=0,keepdims = True).astype(np.int64)
#     train_data -= mean_image
#     train_val_data -= mean_image
#     test_data -= mean_image
# 
#     train_data = np.reshape(train_data,(num_train,1,28,28))
#     test_data = np.reshape(test_data,(num_test,1,28,28))
#     train_val_data = np.reshape(train_val_data,(num_val,1,28,28))
# 
#     return{
#       'train_data':train_data,'train_labels':train_labels,
#       'train_val_data':train_val_data,'train_val_labels':train_val_labels,
#       'test_data':test_data
#     }
# 
# data = get_mnist_data()
# for k, v in data.iteritems():
#     print '%s: ' % k, v.shape
#     
#     
#     
#==============================================================================
'''
import lmdb
import numpy as np
import pandas as pd 
import caffe
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

solver_proto = '/home/xiaoyi/data/MNIST/bvlc_alexnet/lenet_solver.prototxt'
path = '/home/xiaoyi/data/MNIST/'
train_path = path+'train.csv'
test_path = path + 'test.csv'
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
train_np = train_df.values
y = train_np[:,0]
x = train_np[:,1:]
x = x.reshape((x.shape[0],1,28,28))


def covert_lmdb(x,y,path):
    m=x.shape[0]
    map_size=x.nbytes*10#donot worry , mapsize no harm
    # http://lmdb.readthedocs.io/en/release/#environment-class
    env=lmdb.open(path,map_size=map_size)
    # http://lmdb.readthedocs.io/en/release/#lmdb.Transaction
    with env.begin(write=True) as txn:
        for i in range(m):
            datum=caffe.proto.caffe_pb2.Datum()
            datum.channels=x.shape[1]
            datum.height=x.shape[2]
            datum.width=x.shape[3]
            datum.data=x[i].tostring()#tobeytes if np.version.version >1.9
            datum.label=int(y[i])
            str_id='{:08}'.format(i)
            txn.put(str_id.encode('ascii'),datum.SerializeToString())
            
train_lmdb_path=path+'train_lmdb'
test_score_lmdb_path=path+'test_score_lmdb'

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=33)
covert_lmdb(x_train,y_train,train_lmdb_path)
covert_lmdb(x_test,y_test,test_score_lmdb_path)

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver(solver_proto)
solver.solve()

'''

import caffe
import pandas as pd
import numpy as np
import os,sys
from PIL import Image
from skimage import io

path = '/home/xiaoyi/data/MNIST/'
#caffe_root = '/home/xiaoyi/data/MNIST/'
#os.chdir(caffe_root)



#train

solver_proto  = '/home/xiaoyi/data/MNIST/bvlc_alexnet/lenet_solver.prototxt'
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver(solver_proto)
solver.solve()



#test
'''
deply = path + 'bvlc_alexnet/lenet.prototxt'
caffe_model = path +'MNIST_iter_10000.caffemodel'
img = path +'image/test/TestImage_1.bmp'
labels_filename = path +'labels.txt'

net = caffe.Net(deply,caffe_model,caffe.TEST)

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1))
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data',(2,1,0))

im = caffe.io.load_image(img)
#im=io.imread(img)
#im1=np.reshape(im,(28,28,1))
net.blobs['data'].data[...] = transformer.preprocess('data',im)

out =net.forward()
labels = np.loadtxt(labels_filename,str,delimiter='\t')
prob = net.blobs['prob'].data[0].flatten()
print prob
order=prob.argsort()[-1]
print 'the class is :',labels[order]
'''

##predict
#test_path = path+'test.csv'
#labels_filename = path +'labels.txt'
#deply = path +'bvlc_alexnet/lenet.prototxt'
#caffe_model = path+'MNIST_iter_10000.caffemodel'
#caffe.set_device(0)
#caffe.set_mode_gpu()
#net = caffe.Net(deply,caffe_model,caffe.TEST)
#
#transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
#transformer.set_transpose('data',(2,0,1))
#transformer.set_raw_scale('data',255)
#transformer.set_channel_swap('data',(2,1,0))
#
#test = pd.read_csv(test_path)
#test_array = test.values
#n = np.shape(test_array)[0]
#labels = np.loadtxt(labels_filename,str,delimiter='\t')
#a=[]
#for i in xrange(n):
#    img=test_array[1,:].reshape(28,28)
#    img = np.uint8(img)
#    img1 = Image.fromarray(img)
#    img1.save(path+'test_kaggle/1.bmp')
#    im = caffe.io.load_image(img1)
#    net.blob['data'].data[...] = transformer.preprocess('data',im)
#    
#out =net.forward()
#prob = net.blobs['prob'].data[0].flatten()
#order = prob.argsort()[-1]
#a.append(labels[order])
#    
#ImageId = [i+1 for i in range(n)]
#submit_pd = pd.DataFrame({'ImageId':ImageId,'Label':a})
#submit_pd.to_csv(path+'model.csv',index=False)
#

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

