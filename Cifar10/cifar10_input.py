# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:42:11 2017

@author: xiaoyi
"""
import tensorflow as tf
import numpy as np
import os

#%% read data

def read_cifar10(data_dir,is_train,batch_size,shuffle):
    img_width =32
    img_height =32
    img_depth = 3
    label_bytes = 1
    image_bytes =img_width*img_height*img_depth
    
    with tf.name_scope('input'):
        if is_train:
            filenames = [os.path.join(data_dir,'data_batch_%d.bin' %ii)
                                        for ii in np.arange(1,6)]
        else:
            filenames = [os.path.join(data_dir,'test_batch.bin')]
        
        filenames_queue = tf.train.string_input_producer(filenames)
        reader = tf.FixedLengthRecordReader(label_bytes+image_bytes)
        key,value = reader.read(filenames_queue)
        
        record_bytes = tf.decode_raw(value,tf.uint8)
        
        label = tf.slice(record_bytes,[0],[label_bytes])
        label = tf.cast(label,tf.int32)
        
        img_raw = tf.slice(record_bytes,[label_bytes],[image_bytes])
        img_raw = tf.reshape(img_raw,[img_depth,img_height,img_width])
        img = tf.transpose(img_raw,(1,2,0)) #convert from D/H/W to H/W/D
        img = tf.cast(img,tf.float32) 
        
#        data argumentation
#        img = tf.random_crop(img,[24,24,3])
#        img = tf.image.random_flip_left_right(img)
#        img = tf.image.random_brightness(img,max_delta=63)
#        img = tf.image.random_contrast(img,lower=0.2,upper=1.8)
        
#        img = tf.image.per_image_standardization(img)
        
        if shuffle:
            img,label_batch = tf.train.shuffle_batch([img,label],
                                                      batch_size=batch_size,
                                                      num_threads=16,
                                                      capacity=2000,
                                                      min_after_dequeue=1500)
        else:
            img,label_batch = tf.train.batch([img,label],
                                             batch_size=batch_size,
                                             num_threads=16,
                                             capacity=2000)
                                             
        return img,tf.reshape(label_batch,[batch_size])
            
            
            
            
            
##  ONE-HOT Encoding
#        n_classes = 10
#        label_batch = tf.one_hot(label_batch,depth=n_classes)
#        return img,tf.reshape(label_batch,[batch_size,n_classes])
#        
#TEST

#import matplotlib.pyplot as plt
#data_dir = '/home/xiaoyi/data/Cifar10/data/cifar-10-batches-bin/'
#BATCH_SIZE = 2
#img_batch,label_batch = read_cifar10(data_dir,
#                                     is_train=True,
#                                     batch_size=BATCH_SIZE,
#                                     shuffle=True)
#                            
#
#
#with tf.Session() as sess:
#    i =0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    
#    try:
#        while not coord.should_stop() and i<1:
#            img,label = sess.run([img_batch,label_batch])
#            
#            for j in np.arange(BATCH_SIZE):
#                print('label:%d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#            
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    
#    finally:
#        coord.request_stop()
#    coord.join(threads)
#

        
        
        
        
            
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
